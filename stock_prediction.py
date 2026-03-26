import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# 读取数据
print("📊 加载数据...")
all_data = []
for file in ['NVDA_processed.csv', 'TSLA_processed.csv', 'AAPL_processed.csv', 
             'BABA_processed.csv', 'MU_processed.csv']:
    if os.path.exists(f'data/stocks/{file}'):
        df = pd.read_csv(f'data/stocks/{file}')
        all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
combined['date'] = pd.to_datetime(combined['date'])

print(f"✅ 加载完成: {len(combined)} 条记录")

# 特征工程
def create_features(df):
    """创建技术指标特征"""
    df = df.sort_values('date').reset_index(drop=True)
    
    # 滞后特征
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    # 移动平均特征
    for window in [5, 10, 20, 60]:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'std_{window}'] = df['close'].rolling(window=window).std()
    
    # 价格变化特征
    df['price_change_1d'] = df['close'].pct_change(1)
    df['price_change_5d'] = df['close'].pct_change(5)
    df['price_change_10d'] = df['close'].pct_change(10)
    
    # 技术指标
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    
    # 目标变量：未来1天、5天、10天的价格变化
    df['target_1d'] = df['close'].shift(-1) / df['close'] - 1
    df['target_5d'] = df['close'].shift(-5) / df['close'] - 1
    df['target_10d'] = df['close'].shift(-10) / df['close'] - 1
    
    return df.dropna()

def calculate_rsi(prices, window=14):
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 为每只股票创建特征
print("\n🔧 特征工程...")
features_list = []
for symbol in combined['symbol'].unique():
    df_symbol = combined[combined['symbol'] == symbol].copy()
    df_featured = create_features(df_symbol)
    df_featured['symbol'] = symbol
    features_list.append(df_featured)

data = pd.concat(features_list, ignore_index=True)
print(f"✅ 特征创建完成: {len(data)} 条记录")

# 选择特征列
feature_cols = [col for col in data.columns if 'lag' in col or 'ma_' in col or 
                'std_' in col or 'rsi' in col or 'macd' in col or 
                'price_change' in col and 'target' not in col]

print(f"\n📋 使用特征: {len(feature_cols)} 个")
print(feature_cols[:10], "...")

# 训练模型
def train_and_predict(symbol, target_col='target_1d'):
    """为单只股票训练模型"""
    df_stock = data[data['symbol'] == symbol].copy()
    
    if len(df_stock) < 50:
        return None
    
    X = df_stock[feature_cols]
    y = df_stock[target_col]
    
    # 时间序列分割（不用随机分割）
    split_idx = int(len(df_stock) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 训练多个模型
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 计算方向准确率（涨/跌预测正确率）
        direction_actual = np.sign(y_test)
        direction_pred = np.sign(y_pred)
        direction_accuracy = np.mean(direction_actual == direction_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'direction_acc': direction_accuracy,
            'predictions': y_pred,
            'actual': y_test
        }
    
    return results, X_test, y_test

# 为每只股票预测
print("\n🤖 训练预测模型...")
all_results = {}

for symbol in ['NVDA', 'TSLA', 'AAPL', 'BABA', 'MU']:
    print(f"\n{'='*50}")
    print(f"📈 {symbol} 预测分析")
    print(f"{'='*50}")
    
    result = train_and_predict(symbol, target_col='target_1d')
    if result is None:
        print(f"❌ {symbol} 数据不足")
        continue
    
    results, X_test, y_test = result
    all_results[symbol] = results
    
    # 显示结果
    print(f"\n{'模型':<20} {'MSE':<10} {'MAE':<10} {'R²':<10} {'方向准确率':<12}")
    print("-" * 70)
    
    best_model = None
    best_acc = 0
    
    for name, res in results.items():
        print(f"{name:<20} {res['mse']:<10.6f} {res['mae']:<10.6f} {res['r2']:<10.4f} {res['direction_acc']*100:<11.2f}%")
        
        if res['direction_acc'] > best_acc:
            best_acc = res['direction_acc']
            best_model = name
    
    print(f"\n🏆 最佳模型: {best_model} (方向准确率: {best_acc*100:.2f}%)")

# 可视化
print("\n📊 生成可视化...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('AI Stock Price Prediction (Random Forest)', fontsize=16, fontweight='bold')

symbols = ['NVDA', 'TSLA', 'AAPL', 'BABA', 'MU']
for idx, symbol in enumerate(symbols):
    if symbol not in all_results:
        continue
    
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    res = all_results[symbol]['Random Forest']
    actual = res['actual'].values * 100  # 转为百分比
    pred = res['predictions'] * 100
    
    # 绘制实际 vs 预测
    x = range(len(actual))
    ax.plot(x, actual, label='Actual', color='blue', alpha=0.7)
    ax.plot(x, pred, label='Predicted', color='red', alpha=0.7, linestyle='--')
    
    ax.set_title(f'{symbol} (Acc: {res["direction_acc"]*100:.1f}%)', fontsize=11)
    ax.set_ylabel('Return (%)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# 删除多余的子图
if len(symbols) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig('stock_prediction.png', dpi=150, bbox_inches='tight')
print("✅ 图表保存: stock_prediction.png")

plt.show()

# 总结报告
print("\n" + "="*70)
print("🎯 AI 预测模型总结报告")
print("="*70)

print(f"\n{'股票':<8} {'最佳模型':<20} {'方向准确率':<12} {'R²':<10} {'预测能力':<15}")
print("-" * 70)

for symbol, results in all_results.items():
    best = max(results.items(), key=lambda x: x[1]['direction_acc'])
    model_name, res = best
    
    ability = "强" if res['direction_acc'] > 0.55 else "中等" if res['direction_acc'] > 0.5 else "弱"
    
    print(f"{symbol:<8} {model_name:<20} {res['direction_acc']*100:<11.2f}% {res['r2']:<10.4f} {ability:<15}")

print("\n💡 关键发现:")
print("  - 股价预测是极具挑战的任务，准确率>55%已算不错")
print("  - 结合技术指标可以提高预测能力")
print("  - 建议用模型作为参考，而非唯一决策依据")

# 明日预测
print("\n🔮 明日涨跌预测（基于最新数据）:")
for symbol in symbols:
    if symbol in all_results:
        df_latest = data[data['symbol'] == symbol].iloc[-1:]
        if len(df_latest) > 0:
            X_latest = df_latest[feature_cols]
            model = all_results[symbol]['Random Forest']['model']
            pred = model.predict(X_latest)[0]
            direction = "📈 涨" if pred > 0 else "📉 跌" if pred < 0 else "➡️ 平"
            print(f"  {symbol}: {direction} (预测变化: {pred*100:+.2f}%)")

print("\n✅ 分析完成!")