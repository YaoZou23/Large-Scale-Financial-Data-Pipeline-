#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("📊 加载数据...")
all_data = []
data_dir = 'data/stocks'
if not os.path.exists(data_dir):
    data_dir = '.'  # 尝试当前目录

for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(f'{data_dir}/{file}')
        all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
combined['date'] = pd.to_datetime(combined['date'])

print(f"✅ 加载完成: {len(combined)} 条记录")

# 颜色方案
colors = {
    'NVDA': '#76B900',
    'TSLA': '#CC0000', 
    'AAPL': '#555555',
    'BABA': '#FF6600',
    'MU': '#0066CC'
}

def predict_direction(df, days=7):
    """
    修复后的方向预测 - 合理的置信度计算
    """
    if len(df) < 20:
        return 0, 0.3, 0  # 默认横盘，低置信度
    
    df = df.sort_values('date').copy()
    close = df['close'].values
    
    # 1. 多时间框架动量
    mom_3d = (close[-1] - close[-4]) / close[-4] if len(close) >= 4 else 0
    mom_7d = (close[-1] - close[-8]) / close[-8] if len(close) >= 8 else 0
    mom_14d = (close[-1] - close[-15]) / close[-15] if len(close) >= 15 else 0
    
    # 2. 趋势强度 (R²拟合度)
    x = np.arange(min(14, len(close)))
    y = close[-len(x):]
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 3. 波动率 (高波动 = 低置信度)
    volatility = df['daily_return'].tail(14).std()
    vol_factor = max(0.2, 1 - volatility * 10)  # 波动率惩罚
    
    # 4. 综合得分 (-1 到 1)
    momentum_score = (mom_3d * 0.5 + mom_7d * 0.3 + mom_14d * 0.2)
    trend_score = np.sign(slope) * r_squared * 0.3  # 趋势贡献
    
    final_score = np.clip(momentum_score * 0.7 + trend_score, -0.1, 0.1)
    
    # 5. 方向判断
    threshold = 0.008  # 0.8%阈值
    if final_score > threshold:
        direction = 1
    elif final_score < -threshold:
        direction = -1
    else:
        direction = 0
    
    # 6. 修复后的置信度 (永远不会是100%)
    base_confidence = min(0.85, r_squared * vol_factor * 0.8 + 0.2)
    # 方向明确时置信度略高，但不超85%
    if direction != 0:
        confidence = min(0.85, base_confidence * 1.1)
    else:
        confidence = base_confidence * 0.8
    
    return direction, confidence, final_score

def generate_forecast(df, days=7):
    """生成带合理置信区间的预测"""
    direction, conf, score = predict_direction(df)
    
    last_date = df['date'].iloc[-1]
    last_price = df['close'].iloc[-1]
    
    # 基于历史波动率生成区间
    returns = df['daily_return'].tail(20)
    vol = returns.std()
    mean_ret = returns.mean()
    
    # 根据方向调整漂移
    if direction == 1:
        drift = max(mean_ret, vol * 0.3)
    elif direction == -1:
        drift = min(mean_ret, -vol * 0.3)
    else:
        drift = mean_ret * 0.5
    
    forecasts = []
    current = last_price
    
    for i in range(1, days + 1):
        date = last_date + timedelta(days=i)
        # 随机游走 + 漂移
        change = np.random.normal(drift, vol * 0.7)
        current *= (1 + change)
        
        # 置信区间随时间扩大
        uncertainty = vol * np.sqrt(i) * 1.5
        upper = current * (1 + uncertainty)
        lower = current * (1 - uncertainty)
        
        forecasts.append({
            'date': date,
            'mid': current,
            'upper': upper,
            'lower': lower
        })
    
    return {
        'direction': direction,
        'confidence': conf,
        'score': score,
        'forecasts': forecasts,
        'last_price': last_price
    }

# ==================== 创建图表 ====================
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

fig.suptitle('AI Stock Analysis with 7-Day Direction Forecast\n(2025.06 - 2026.03)', 
             fontsize=18, fontweight='bold', y=0.98)

# 存储预测结果
all_predictions = []

# ==================== 1. 价格走势 + 7天预测 (左上) ====================
ax1 = fig.add_subplot(gs[0, 0])

for symbol in ['NVDA', 'TSLA', 'AAPL', 'BABA', 'MU']:
    df = combined[combined['symbol'] == symbol].sort_values('date')
    if len(df) < 30:
        continue
    
    # 标准化价格
    normalized = df['close'] / df['close'].iloc[0] * 100
    color = colors.get(symbol, 'gray')
    
    # 绘制历史走势
    ax1.plot(df['date'], normalized, label=symbol, color=color, 
             linewidth=2.5, alpha=0.9)
    
    # 生成预测
    forecast = generate_forecast(df, days=7)
    all_predictions.append({
        'symbol': symbol,
        **forecast
    })
    
    # 绘制预测区间
    last_date = df['date'].iloc[-1]
    last_norm = normalized.iloc[-1]
    
    future_dates = [f['date'] for f in forecast['forecasts']]
    mid_prices = [f['mid'] / df['close'].iloc[0] * 100 for f in forecast['forecasts']]
    upper_prices = [f['upper'] / df['close'].iloc[0] * 100 for f in forecast['forecasts']]
    lower_prices = [f['lower'] / df['close'].iloc[0] * 100 for f in forecast['forecasts']]
    
    # 填充置信区间
    ax1.fill_between(future_dates, lower_prices, upper_prices, 
                     alpha=0.15, color=color)
    # 预测中线
    ax1.plot([last_date] + future_dates, [last_norm] + mid_prices, 
             color=color, linestyle='--', linewidth=2, alpha=0.7)

# 添加预测区域标注
ax1.axvline(x=all_predictions[0]['forecasts'][0]['date'], 
            color='red', linestyle=':', alpha=0.5, linewidth=1.5)
ax1.text(0.75, 0.95, 'Forecast\nRegion', transform=ax1.transAxes,
         fontsize=9, color='red', alpha=0.7, ha='center',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.set_title('📈 Price Trend & 7-Day Forecast\n(Normalized, Shaded=Confidence Band)', 
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Date', fontsize=10)
ax1.set_ylabel('Normalized Price', fontsize=10)
ax1.legend(loc='upper left', fontsize=9, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ==================== 2. 方向预测指示器 (右上) ====================
ax2 = fig.add_subplot(gs[0, 1])

direction_emoji = {1: "📈", -1: "📉", 0: "➡️"}
direction_text = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
direction_color = {1: "#00C851", -1: "#ff4444", 0: "#ffbb33"}

symbols = [p['symbol'] for p in all_predictions]
directions = [p['direction'] for p in all_predictions]
confidences = [p['confidence'] * 100 for p in all_predictions]

# 绘制置信度条形图
bar_colors = [direction_color[d] for d in directions]
bars = ax2.barh(symbols, confidences, color=bar_colors, edgecolor='black', alpha=0.8)

# 添加方向标签
for i, (bar, sym, direc, conf) in enumerate(zip(bars, symbols, directions, confidences)):
    width = bar.get_width()
    emoji = direction_emoji[direc]
    text = direction_text[direc]
    ax2.text(width + 3, bar.get_y() + bar.get_height()/2,
             f'{emoji} {text}', ha='left', va='center', fontsize=11, fontweight='bold')

ax2.set_xlim(0, 100)
ax2.set_title('🎯 7-Day Direction Forecast\n(Confidence Score)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Confidence (%)', fontsize=10)

# 添加说明
ax2.text(0.02, 0.02, 'Based on: Multi-frame momentum + Trend strength + Volatility',
         transform=ax2.transAxes, fontsize=8, alpha=0.6)

# ==================== 3. 预测明细表 (左中) ====================
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')

# 创建预测表格
table_data = []
for p in all_predictions:
    direction = p['direction']
    conf = p['confidence']
    last_p = p['last_price']
    future_p = p['forecasts'][-1]['mid']
    change = (future_p / last_p - 1) * 100
    
    table_data.append([
        p['symbol'],
        direction_emoji[direction] + " " + direction_text[direction],
        f"{conf*100:.1f}%",
        f"${last_p:.2f}",
        f"${future_p:.2f}",
        f"{change:+.1f}%"
    ])

table = ax3.table(
    cellText=table_data,
    colLabels=['Symbol', 'Direction', 'Confidence', 'Current', '7D Target', 'Expected Change'],
    cellLoc='center',
    loc='center',
    colWidths=[0.12, 0.2, 0.15, 0.15, 0.15, 0.15]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# 颜色编码
for i, p in enumerate(all_predictions):
    table[(i+1, 1)].set_facecolor(direction_color[p['direction']])
    table[(i+1, 1)].set_text_props(weight='bold')

ax3.set_title('📋 Forecast Details', fontsize=12, fontweight='bold', pad=20)

# ==================== 4. 风险-动量矩阵 (右中) ====================
ax4 = fig.add_subplot(gs[1, 1])

for p in all_predictions:
    symbol = p['symbol']
    df = combined[combined['symbol'] == symbol]
    
    # X轴: 波动率 (风险)
    volatility = df['daily_return'].tail(20).std() * 100
    
    # Y轴: 动量得分
    momentum = p['score'] * 1000  # 放大显示
    
    color = colors.get(symbol, 'gray')
    ax4.scatter(volatility, momentum, s=400, c=color, 
               edgecolors='black', linewidths=2, alpha=0.8, zorder=5)
    
    # 添加标签
    ax4.annotate(symbol, (volatility, momentum),
                xytext=(7, 7), textcoords='offset points',
                fontsize=12, fontweight='bold')

ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('Volatility (Risk) %', fontsize=10)
ax4.set_ylabel('Momentum Score', fontsize=10)
ax4.set_title('⚡ Risk-Momentum Matrix\n(Upper Left = High Momentum, Low Risk)', 
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 添加象限标签
ax4.text(0.95, 0.95, 'High Mom\nLow Risk\n★', transform=ax4.transAxes,
        fontsize=9, ha='right', va='top', color='green', fontweight='bold')

# ==================== 5. 技术指标热力图 (底部) ====================
ax5 = fig.add_subplot(gs[2, :])

tech_data = []
for symbol in ['NVDA', 'TSLA', 'AAPL', 'BABA', 'MU']:
    df = combined[combined['symbol'] == symbol].sort_values('date')
    if len(df) < 20:
        continue
    
    # 计算各项技术指标
    close = df['close']
    
    # 均线排列
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma_trend = 100 if close.iloc[-1] > ma5 > ma20 else \
               60 if close.iloc[-1] > ma5 else \
               30 if close.iloc[-1] > ma20 else 0
    
    # MACD
    ema12 = close.ewm(span=12).mean().iloc[-1]
    ema26 = close.ewm(span=26).mean().iloc[-1]
    macd = ema12 - ema26
    macd_score = min(100, max(0, (macd / close.iloc[-1] * 100 + 5) * 10))
    
    # RSI简化版
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
    rs = gain / loss if loss != 0 else 1
    rsi = 100 - (100 / (1 + rs))
    rsi_score = 100 - abs(rsi - 50) * 2  # 距离50越远分数越高(趋势强)
    
    # 预测方向匹配度
    pred = next((p for p in all_predictions if p['symbol'] == symbol), None)
    forecast_score = pred['confidence'] * 100 if pred else 50
    
    tech_data.append({
        'Symbol': symbol,
        'MA Trend': ma_trend,
        'MACD': macd_score,
        'RSI Strength': rsi_score,
        'Forecast Conf': forecast_score,
        'Momentum': min(100, max(0, df['daily_return'].tail(5).sum() * 500 + 50))
    })

tech_df = pd.DataFrame(tech_data).set_index('Symbol')
tech_df = tech_df.reindex(['NVDA', 'TSLA', 'AAPL', 'BABA', 'MU'])

im = ax5.imshow(tech_df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax5.set_xticks(range(len(tech_df.columns)))
ax5.set_xticklabels(tech_df.columns, fontsize=11)
ax5.set_yticks(range(len(tech_df.index)))
ax5.set_yticklabels(tech_df.index, fontsize=11)

for i in range(len(tech_df.index)):
    for j in range(len(tech_df.columns)):
        val = tech_df.iloc[i, j]
        color = "white" if val < 30 or val > 70 else "black"
        ax5.text(j, i, f'{val:.0f}', ha="center", va="center", 
                color=color, fontsize=12, fontweight='bold')

ax5.set_title('📊 Technical Indicator Heatmap\n(Forecast Conf = Model Confidence in Prediction)', 
              fontsize=12, fontweight='bold', pad=20)

cbar = plt.colorbar(im, ax=ax5, orientation='horizontal', pad=0.15, aspect=40)
cbar.set_label('Score (0-100)', fontsize=10)

# ==================== 保存 ====================
plt.tight_layout()
plt.savefig('advanced_stock_dashboard.png', dpi=200, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("\n✅ 可视化已保存: advanced_stock_dashboard.png")

# ==================== 打印预测结果 ====================
print("\n" + "="*70)
print("🔮 未来7天方向预测 (修复版)")
print("="*70)

for p in all_predictions:
    symbol = p['symbol']
    direc = p['direction']
    conf = p['confidence']
    last_p = p['last_price']
    future_p = p['forecasts'][-1]['mid']
    change = (future_p / last_p - 1) * 100
    
    print(f"\n{symbol}: {direction_emoji[direc]} {direction_text[direc]} "
          f"(置信度: {conf*100:.1f}%)")
    print(f"   当前: ${last_p:.2f} → 7天预测: ${future_p:.2f} ({change:+.1f}%)")
    print(f"   预测区间: ${p['forecasts'][-1]['lower']:.2f} - ${p['forecasts'][-1]['upper']:.2f}")

print("\n" + "="*70)
print("⚠️  免责声明: 此预测仅基于历史动量，不构成投资建议")
print("="*70)