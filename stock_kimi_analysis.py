import json
import os
from datetime import datetime

import pandas as pd
import requests

# Kimi API 配置
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "sk-D0fxRM2rrN4QuLWAhBc1ZQdyFEPxdk6bgpPfkzIzTrSF12IE")
KIMI_API_URL = "https://api.moonshot.cn/v1/chat/completions"

# 读取股票数据
print("📊 加载股票数据...")
all_data = []
for file in os.listdir("data/stocks"):
    if file.endswith(".csv"):
        df = pd.read_csv(f"data/stocks/{file}")
        all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)


# 计算股票指标
def get_stock_summary(symbol):
    df = combined[combined["symbol"] == symbol].sort_values("date")
    if len(df) == 0:
        return None

    latest = df.iloc[-1]
    start_price = df["close"].iloc[0]
    end_price = df["close"].iloc[-1]

    # 计算额外指标
    df["high_20d"] = df["high"].rolling(20).max()
    df["low_20d"] = df["low"].rolling(20).min()

    return {
        "symbol": symbol,
        "name": latest["name"],
        "分析周期": f"{df['date'].iloc[0]} 至 {df['date'].iloc[-1]}",
        "起始价格": f"${round(start_price, 2)}",
        "当前价格": f"${round(end_price, 2)}",
        "总收益率": f"{round((end_price - start_price) / start_price * 100, 2)}%",
        "价格区间": f"${round(df['close'].min(), 2)} - ${round(df['close'].max(), 2)}",
        "平均成交量": f"{round(df['volume'].mean() / 1e6, 2)}M",
        "波动率": f"{round(df['daily_return'].std() * 100, 2)}%",
        "技术指标": {
            "MA5": round(latest["ma5"], 2),
            "MA20": round(latest["ma20"], 2),
            "MA60": round(latest["ma60"], 2),
            "MACD": round(latest["macd"], 2),
            "MACD信号": round(latest["macd_signal"], 2),
        },
        "近期趋势": "上涨📈"
        if latest["close"] > latest["ma5"] > latest["ma20"]
        else "下跌📉"
        if latest["close"] < latest["ma5"] < latest["ma20"]
        else "震荡↔️",
        "最近5天涨跌": f"{round(df['daily_return'].tail(5).sum() * 100, 2)}%",
    }


# 生成 Prompt
def generate_prompt(summaries):
    prompt = f"""你是一位专业的股票分析师，拥有10年量化投资经验。请基于以下美股数据进行深度分析：

## 📊 数据概览
分析时间：{datetime.now().strftime('%Y年%m月%d日')}
数据周期：{summaries[0]['分析周期']}

## 📈 各股票详细数据
"""

    for s in summaries:
        prompt += f"""
### {s['symbol']} - {s['name']}
- **价格走势**: {s['起始价格']} → {s['当前价格']} ({s['总收益率']})
- **价格波动区间**: {s['价格区间']}
- **波动率**: {s['波动率']}
- **成交量**: {s['平均成交量']}
- **技术指标**: MA5={s['技术指标']['MA5']}, MA20={s['技术指标']['MA20']}, MACD={s['技术指标']['MACD']}
- **近期趋势**: {s['近期趋势']}
- **最近5天表现**: {s['最近5天涨跌']}
"""

    prompt += """
## 🎯 请提供以下专业分析：

### 1. 技术面评分（每只股票）
- 趋势强度（1-10分）
- 支撑/阻力位
- 买入/卖出信号

### 2. 投资价值排名
- 最值得投资的1只股票及理由
- 最应避免的1只股票及理由

### 3. 风险评估
- 每只股票的主要风险点
- 整体市场风险

### 4. 未来1个月预测
- 价格目标区间
- 关键时间节点

### 5. 投资组合建议
- 5只股票的最佳配置比例
- 调仓策略

请用专业、客观的语言，结合技术指标给出建议。注意：这只是技术分析，不构成投资建议。"""

    return prompt


# 获取所有股票数据
print("🔧 生成分析报告...")
summaries = []
for symbol in ["NVDA", "TSLA", "AAPL", "BABA", "MU"]:
    summary = get_stock_summary(symbol)
    if summary:
        summaries.append(summary)

# 生成 Prompt
prompt = generate_prompt(summaries)

print("\n" + "=" * 70)
print("📝 生成的 Prompt（预览）")
print("=" * 70)
print(prompt[:500] + "...")
print("=" * 70)

# 保存 prompt
with open("kimi_prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)
print("\n✅ Prompt 已保存: kimi_prompt.txt")


# 调用 Kimi API
def call_kimi_api(prompt):
    headers = {"Authorization": f"Bearer {KIMI_API_KEY}", "Content-Type": "application/json"}

    data = {
        "model": "moonshot-v1-8k",
        "messages": [
            {"role": "system", "content": "你是一位专业的股票量化分析师，擅长技术分析和投资策略。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 4000,
    }

    print("\n🤖 调用 Kimi API...")
    response = requests.post(KIMI_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"❌ API 错误: {response.status_code}")
        print(response.text)
        return None


# 执行分析
if KIMI_API_KEY != "your-api-key-here":
    analysis = call_kimi_api(prompt)

    if analysis:
        print("\n" + "=" * 70)
        print("📊 Kimi AI 股票分析报告")
        print("=" * 70)
        print(analysis)

        # 保存结果
        with open("kimi_analysis_report.md", "w", encoding="utf-8") as f:
            f.write("# 🚀 Kimi AI 股票深度分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("**分析模型**: Moonshot-v1-8k\n\n")
            f.write("---\n\n")
            f.write(analysis)

        print("\n" + "=" * 70)
        print("✅ 完整报告已保存: kimi_analysis_report.md")
        print("=" * 70)
else:
    print("\n⚠️ 请设置 Kimi API Key:")
    print("export KIMI_API_KEY='your-api-key-here'")
    print("\n或手动复制 kimi_prompt.txt 到 Kimi 网页版分析")

import pandas as pd
import os
import requests
import json
from datetime import datetime

# Kimi API 配置
KIMI_API_KEY = os.getenv('KIMI_API_KEY', 'sk-7ugd2wlPEqradGOEf6lnAz5j94Zp1yT59nczF50Nnn4BKtXD')
KIMI_API_URL = "https://api.moonshot.cn/v1/chat/completions"

# 读取股票数据
print("📊 加载股票数据...")
all_data = []
for file in os.listdir('data/stocks'):
    if file.endswith('.csv'):
        df = pd.read_csv(f'data/stocks/{file}')
        all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)

# 计算股票指标
def get_stock_summary(symbol):
    df = combined[combined['symbol'] == symbol].sort_values('date')
    if len(df) == 0:
        return None
    
    latest = df.iloc[-1]
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    
    # 计算额外指标
    df['high_20d'] = df['high'].rolling(20).max()
    df['low_20d'] = df['low'].rolling(20).min()
    
    return {
        'symbol': symbol,
        'name': latest['name'],
        '分析周期': f"{df['date'].iloc[0]} 至 {df['date'].iloc[-1]}",
        '起始价格': f"${round(start_price, 2)}",
        '当前价格': f"${round(end_price, 2)}",
        '总收益率': f"{round((end_price - start_price) / start_price * 100, 2)}%",
        '价格区间': f"${round(df['close'].min(), 2)} - ${round(df['close'].max(), 2)}",
        '平均成交量': f"{round(df['volume'].mean()/1e6, 2)}M",
        '波动率': f"{round(df['daily_return'].std() * 100, 2)}%",
        '技术指标': {
            'MA5': round(latest['ma5'], 2),
            'MA20': round(latest['ma20'], 2),
            'MA60': round(latest['ma60'], 2),
            'MACD': round(latest['macd'], 2),
            'MACD信号': round(latest['macd_signal'], 2)
        },
        '近期趋势': '上涨📈' if latest['close'] > latest['ma5'] > latest['ma20'] else 
                   '下跌📉' if latest['close'] < latest['ma5'] < latest['ma20'] else '震荡↔️',
        '最近5天涨跌': f"{round(df['daily_return'].tail(5).sum() * 100, 2)}%"
    }

# 生成 Prompt
def generate_prompt(summaries):
    prompt = f"""你是一位专业的股票分析师，拥有10年量化投资经验。请基于以下美股数据进行深度分析：

## 📊 数据概览
分析时间：{datetime.now().strftime('%Y年%m月%d日')}
数据周期：{summaries[0]['分析周期']}

## 📈 各股票详细数据
"""
    
    for s in summaries:
        prompt += f"""
### {s['symbol']} - {s['name']}
- **价格走势**: {s['起始价格']} → {s['当前价格']} ({s['总收益率']})
- **价格波动区间**: {s['价格区间']}
- **波动率**: {s['波动率']}
- **成交量**: {s['平均成交量']}
- **技术指标**: MA5={s['技术指标']['MA5']}, MA20={s['技术指标']['MA20']}, MACD={s['技术指标']['MACD']}
- **近期趋势**: {s['近期趋势']}
- **最近5天表现**: {s['最近5天涨跌']}
"""
    
    prompt += """
## 🎯 请提供以下专业分析：

### 1. 技术面评分（每只股票）
- 趋势强度（1-10分）
- 支撑/阻力位
- 买入/卖出信号

### 2. 投资价值排名
- 最值得投资的1只股票及理由
- 最应避免的1只股票及理由

### 3. 风险评估
- 每只股票的主要风险点
- 整体市场风险

### 4. 未来1个月预测
- 价格目标区间
- 关键时间节点

### 5. 投资组合建议
- 5只股票的最佳配置比例
- 调仓策略

请用专业、客观的语言，结合技术指标给出建议。注意：这只是技术分析，不构成投资建议。"""

    return prompt

# 获取所有股票数据
print("🔧 生成分析报告...")
summaries = []
for symbol in ['NVDA', 'TSLA', 'AAPL', 'BABA', 'MU']:
    summary = get_stock_summary(symbol)
    if summary:
        summaries.append(summary)

# 生成 Prompt
prompt = generate_prompt(summaries)

print("\n" + "="*70)
print("📝 生成的 Prompt（预览）")
print("="*70)
print(prompt[:500] + "...")
print("="*70)

# 保存 prompt
with open('kimi_prompt.txt', 'w', encoding='utf-8') as f:
    f.write(prompt)
print("\n✅ Prompt 已保存: kimi_prompt.txt")

# 调用 Kimi API
def call_kimi_api(prompt):
    headers = {
        "Authorization": f"Bearer {KIMI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "moonshot-v1-8k",
        "messages": [
            {"role": "system", "content": "你是一位专业的股票量化分析师，擅长技术分析和投资策略。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    
    print("\n🤖 调用 Kimi API...")
    response = requests.post(KIMI_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"❌ API 错误: {response.status_code}")
        print(response.text)
        return None

# 执行分析
if KIMI_API_KEY != 'your-api-key-here':
    analysis = call_kimi_api(prompt)
    
    if analysis:
        print("\n" + "="*70)
        print("📊 Kimi AI 股票分析报告")
        print("="*70)
        print(analysis)
        
        # 保存结果
        with open('kimi_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(f"# 🚀 Kimi AI 股票深度分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"**分析模型**: Moonshot-v1-8k\n\n")
            f.write("---\n\n")
            f.write(analysis)
        
        print("\n" + "="*70)
        print("✅ 完整报告已保存: kimi_analysis_report.md")
        print("="*70)
else:
    print("\n⚠️ 请设置 Kimi API Key:")
    print("export KIMI_API_KEY='your-api-key-here'")
    print("\n或手动复制 kimi_prompt.txt 到 Kimi 网页版分析")
