#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('MPS' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ 使用设备: {device}")

# ==================== 1. 数据集定义 ====================

class StockDataset(Dataset):
    """PyTorch数据集"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ==================== 2. PyTorch模型架构 ====================

class LSTMPredictor(nn.Module):
    """LSTM方向预测模型"""
    def __init__(self, input_size=18, hidden_size=64, num_layers=2, num_classes=3, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)
        
        # LSTM输出
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden)
        
        # 注意力加权
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden)
        
        # 分类
        output = self.classifier(context)
        return output, attention_weights

class TransformerPredictor(nn.Module):
    """Transformer时间序列预测"""
    def __init__(self, input_size=18, d_model=64, nhead=4, num_layers=2, num_classes=3, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # Transformer编码 (使用最后一个时间步)
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # 取平均池化
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        output = self.classifier(pooled)
        return output

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

# ==================== 3. 特征工程 (为深度学习优化) ====================

def create_sequences(df, seq_length=20, predict_days=7):
    """
    创建时间序列样本
    返回: X(序列), y(标签)
    """
    df = df.sort_values('date').copy()
    
    # 特征工程 (与之前类似，但归一化)
    features = pd.DataFrame()
    features['returns_1d'] = df['close'].pct_change()
    features['returns_5d'] = df['close'].pct_change(5)
    features['volatility'] = features['returns_1d'].rolling(20).std()
    features['ma_ratio'] = df['close'].rolling(5).mean() / df['close'].rolling(20).mean()
    features['momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    features['rsi'] = calculate_rsi(df['close'])
    features['macd'] = calculate_macd(df['close'])
    
    # 价格位置 (0-1归一化)
    rolling_max = df['close'].rolling(60).max()
    rolling_min = df['close'].rolling(60).min()
    features['price_position'] = (df['close'] - rolling_min) / (rolling_max - rolling_min)
    
    # 填充NaN
    features = features.ffill().fillna(0)
    
    # 创建序列
    X, y = [], []
    for i in range(len(features) - seq_length - predict_days):
        seq = features.iloc[i:i+seq_length].values
        # 标签: 未来7天方向
        future_return = (df['close'].iloc[i+seq_length+predict_days] / 
                        df['close'].iloc[i+seq_length] - 1)
        
        if future_return > 0.02:
            label = 2  # 涨
        elif future_return < -0.02:
            label = 0  # 跌
        else:
            label = 1  # 平
            
        X.append(seq)
        y.append(label)
    
    return np.array(X), np.array(y)

def calculate_rsi(prices, period=14):
    """计算RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    """计算MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    return ema_fast - ema_slow

# ==================== 4. PyTorch训练流程 ====================

class PyTorchStockTrainer:
    def __init__(self, model_type='lstm', seq_length=20):
        self.model_type = model_type
        self.seq_length = seq_length
        self.model = None
        self.is_trained = False
        
    def build_model(self, input_size=8):
        """构建模型"""
        if self.model_type == 'lstm':
            self.model = LSTMPredictor(input_size=input_size).to(device)
        elif self.model_type == 'transformer':
            self.model = TransformerPredictor(input_size=input_size).to(device)
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")
            
        print(f"✅ 构建模型: {self.model_type}")
        print(f"   参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train(self, combined_df, symbols=['NVDA', 'TSLA', 'AAPL', 'BABA', 'MU'], 
              epochs=500, batch_size=32, lr=0.001):
        """训练模型"""
        print(f"\n🚀 开始训练 {self.model_type} 模型...")
        
        # 准备数据
        all_X, all_y = [], []
        for symbol in symbols:
            df = combined_df[combined_df['symbol'] == symbol].sort_values('date')
            if len(df) < 100:
                continue
                
            X, y = create_sequences(df, self.seq_length)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                print(f"  {symbol}: {len(X)} 个序列")
        
        if not all_X:
            print("❌ 数据不足")
            return False
            
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        # 划分训练/验证集 (时间序列不能用随机划分，这里简化处理)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 数据加载器
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 训练配置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 训练循环
        best_val_acc = 0
        history = {'train_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证
            self.model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs, _ = self.model(batch_X)
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / val_total
            avg_loss = train_loss / len(train_loader)
            
            history['train_loss'].append(avg_loss)
            history['val_acc'].append(val_acc)
            scheduler.step(avg_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f'best_{self.model_type}_model.pth')
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Val Acc={val_acc:.2%}")
        
        self.is_trained = True
        print(f"\n✅ 训练完成! 最佳验证准确率: {best_val_acc:.2%}")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(f'best_{self.model_type}_model.pth'))
        return True
    
    def predict(self, df, days=7):
        """预测"""
        if not self.is_trained:
            return None
            
        self.model.eval()
        
        # 准备最新序列
        X, _ = create_sequences(df, self.seq_length)
        if len(X) == 0:
            return None
            
        latest_seq = torch.FloatTensor(X[-1:]).to(device)  # (1, seq_len, features)
        
        with torch.no_grad():
            output, attention = self.model(latest_seq)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # 方向映射
        classes = ['下跌', '横盘', '上涨']
        pred_idx = np.argmax(probabilities)
        direction = classes[pred_idx]
        confidence = probabilities[pred_idx]
        
        # 生成价格路径 (基于预期收益率)
        last_price = df['close'].iloc[-1]
        last_date = pd.to_datetime(df['date'].iloc[-1]) if 'date' in df.columns else datetime.now()
        
        # 根据预测方向设定漂移
        if pred_idx == 2:  # 涨
            expected_return = 0.03
        elif pred_idx == 0:  # 跌
            expected_return = -0.03
        else:
            expected_return = 0.0
        
        forecasts = []
        current_price = last_price
        daily_vol = df['close'].pct_change().std()
        
        for i in range(1, days + 1):
            date = last_date + timedelta(days=i)
            drift = expected_return / days
            uncertainty = daily_vol * np.sqrt(i) * 1.5
            
            current_price *= (1 + drift + np.random.normal(0, daily_vol * 0.3))
            
            forecasts.append({
                'date': date,
                'mid': current_price,
                'upper': current_price * (1 + uncertainty),
                'lower': current_price * (1 - uncertainty)
            })
        
        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': {cls: prob for cls, prob in zip(classes, probabilities)},
            'expected_return': expected_return,
            'forecasts': forecasts,
            'last_price': last_price,
            'attention_weights': attention.cpu().numpy()[0] if attention is not None else None
        }

# ==================== 5. 主程序 ====================

def main():
    # 读取数据
    print("📊 加载数据...")
    all_data = []
    data_dir = 'data/stocks'
    
    if not os.path.exists(data_dir):
        data_dir = '.'
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(f'{data_dir}/{file}')
            all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])
    print(f"✅ 加载完成: {len(combined)} 条记录")
    
    # 选择模型类型: 'lstm' 或 'transformer'
    model_type = 'lstm'  # 可以改为 'transformer'
    
    # 初始化训练器
    trainer = PyTorchStockTrainer(model_type=model_type, seq_length=20)
    trainer.build_model(input_size=8)  # 8个特征
    
    # 训练
    success = trainer.train(combined, epochs=500, batch_size=16)
    
    if not success:
        return
    
    # 预测
    symbols = ['NVDA', 'TSLA', 'AAPL', 'BABA', 'MU']
    predictions = {}
    
    print("\n" + "="*70)
    print(f"🔮 {model_type.upper()} 模型预测结果 (7天)")
    print("="*70)
    
    for symbol in symbols:
        df = combined[combined['symbol'] == symbol].sort_values('date')
        if len(df) < 60:
            continue
            
        pred = trainer.predict(df)
        if pred:
            predictions[symbol] = pred
            
            print(f"\n{symbol}:")
            print(f"  方向: {pred['direction']} (置信度: {pred['confidence']:.1%})")
            print(f"  概率分布: {pred['probabilities']}")
            print(f"  预期收益: {pred['expected_return']:+.1%}")
    
    # 可视化
    visualize(combined, predictions, model_type)
    
    return predictions

def visualize(combined, predictions, model_type):
    """可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = {'上涨': '#00C851', '下跌': '#ff4444', '横盘': '#ffbb33'}
    
    for idx, (symbol, pred) in enumerate(predictions.items()):
        if idx >= 5:
            break
            
        ax = axes[idx]
        df = combined[combined['symbol'] == symbol].sort_values('date').tail(60)
        
        # 历史价格
        ax.plot(df['date'], df['close'], color='gray', alpha=0.7, label='历史')
        
        # 预测
        last_date = df['date'].iloc[-1]
        last_price = df['close'].iloc[-1]
        
        future_dates = [f['date'] for f in pred['forecasts']]
        mids = [f['mid'] for f in pred['forecasts']]
        ups = [f['upper'] for f in pred['forecasts']]
        lows = [f['lower'] for f in pred['forecasts']]
        
        color = colors[pred['direction']]
        ax.fill_between(future_dates, lows, ups, alpha=0.2, color=color)
        ax.plot([last_date] + future_dates, [last_price] + mids, 
                color=color, linestyle='--', linewidth=2.5)
        
        ax.set_title(f"{symbol}: {pred['direction']} ({pred['confidence']:.0%})", 
                    color=color, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 汇总
    ax = axes[-1]
    syms = list(predictions.keys())
    confs = [predictions[s]['confidence'] for s in syms]
    dirs = [predictions[s]['direction'] for s in syms]
    bar_colors = [colors[d] for d in dirs]
    
    ax.barh(syms, confs, color=bar_colors, edgecolor='black')
    ax.set_xlim(0, 1)
    ax.set_title(f'{model_type.upper()} Model Confidence', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{model_type}_predictions.png', dpi=200, bbox_inches='tight')
    print(f"\n✅ 图表已保存: {model_type}_predictions.png")

if __name__ == "__main__":
    predictions = main()