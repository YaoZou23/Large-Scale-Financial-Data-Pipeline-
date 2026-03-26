
#!/usr/bin/env python3
import os
import pandas as pd
import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

# LangChain 1.x 新导入
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ==================== 配置 ====================
KIMI_API_KEY = os.getenv('KIMI_API_KEY', 'sk-mZwAzL8FAq01vIhxB09npuS5sIGYL9xXLnthq8YmFXc3ZtrW')
KIMI_BASE_URL = "https://api.moonshot.cn/v1"

# 定义输出结构（所有字段设为可选，防止缺失）
class StockAnalysis(BaseModel):
    investment_rating: Optional[str] = Field(default="UNKNOWN", description="投资评级")
    confidence_score: Optional[int] = Field(default=5, description="置信度 1-10")
    target_price_1m: Optional[float] = Field(default=0, description="1个月目标价")
    target_price_3m: Optional[float] = Field(default=0, description="3个月目标价")
    key_reasons: Optional[List[str]] = Field(default_factory=list, description="关键理由")
    risk_factors: Optional[List[str]] = Field(default_factory=list, description="风险因素")
    technical_analysis: Optional[str] = Field(default="", description="技术分析")
    suggested_position: Optional[str] = Field(default="HOLD", description="建议仓位")

# 初始化 Kimi
llm = ChatOpenAI(
    model="moonshot-v1-8k",
    api_key=KIMI_API_KEY,
    base_url=KIMI_BASE_URL,
    temperature=0.7,
    max_tokens=2000
)

# ==================== 数据加载 ====================
def load_stock_data():
    all_data = []
    for file in os.listdir('data/stocks'):
        if file.endswith('.csv'):
            df = pd.read_csv(f'data/stocks/{file}')
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def calculate_metrics(df, symbol):
    stock = df[df['symbol'] == symbol].sort_values('date')
    if len(stock) == 0:
        return None
    
    latest = stock.iloc[-1]
    start = stock['close'].iloc[0]
    end = stock['close'].iloc[-1]
    daily_returns = stock['daily_return'].dropna()
    
    return {
        'symbol': symbol,
        'name': latest['name'],
        'period': f"{stock['date'].iloc[0]} 至 {stock['date'].iloc[-1]}",
        'total_return_pct': round((end - start) / start * 100, 2),
        'current_price': round(end, 2),
        'start_price': round(start, 2),
        'volatility_pct': round(daily_returns.std() * 100, 2),
        'sharpe_ratio': round(daily_returns.mean() / daily_returns.std() * (252 ** 0.5), 2) if daily_returns.std() != 0 else 0,
        'trading_days': len(stock),
        'up_days': int((daily_returns > 0).sum()),
        'down_days': int((daily_returns < 0).sum()),
        'ma5': round(latest['ma5'], 2),
        'ma20': round(latest['ma20'], 2),
        'macd': round(latest['macd'], 2),
        'recent_5d_return': round(stock['daily_return'].tail(5).sum() * 100, 2),
        'trend': 'BULL' if end > latest['ma5'] > latest['ma20'] else 'BEAR' if end < latest['ma5'] < latest['ma20'] else 'NEUTRAL'
    }

# ==================== 分析函数（带容错） ====================
def analyze_stock_safe(symbol, metrics):
    """安全地分析股票，处理各种错误"""
    
    prompt_text = f"""
你是一位资深量化投资分析师。请分析以下股票，并以JSON格式返回：

【{metrics['symbol']} - {metrics['name']}】
- 分析周期: {metrics['period']}
- 总收益率: {metrics['total_return_pct']}%
- 当前价: ${metrics['current_price']}
- 波动率: {metrics['volatility_pct']}%
- 夏普比率: {metrics['sharpe_ratio']}
- 趋势: {metrics['trend']}
- MACD: {metrics['macd']}

请返回以下JSON格式（不要有任何其他文字）:
{{
    "investment_rating": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
    "confidence_score": 1-10,
    "target_price_1m": 数字,
    "target_price_3m": 数字,
    "key_reasons": ["理由1", "理由2", "理由3"],
    "risk_factors": ["风险1", "风险2"],
    "technical_analysis": "技术面分析总结",
    "suggested_position": "LIGHT/MEDIUM/HEAVY"
}}
"""
    
    try:
        # 直接调用 LLM，不用结构化输出
        response = llm.invoke(prompt_text)
        content = response.content
        
        # 提取 JSON
        try:
            # 找 JSON 块
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                json_str = content.split('```')[1].split('```')[0]
            else:
                # 找第一个 { 和最后一个 }
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end] if start != -1 and end != 0 else content
            
            data = json.loads(json_str.strip())
            
            # 用 Pydantic 验证和填充默认值
            analysis = StockAnalysis(**data)
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON解析失败: {e}")
            # 返回默认结构
            return StockAnalysis(
                investment_rating="UNKNOWN",
                confidence_score=5,
                technical_analysis=content[:200]  # 保存原始内容
            )
            
    except Exception as e:
        print(f"    ❌ 调用失败: {e}")
        return StockAnalysis(investment_rating="ERROR")

# ==================== 主程序 ====================
def main():
    print("="*70)
    print("🤖 LangChain + Kimi 智能分析（容错版）")
    print("="*70)
    
    df = load_stock_data()
    print(f"✅ 加载: {len(df)} 条记录\n")
    
    results = []
    
    for symbol in ['NVDA', 'TSLA', 'AAPL', 'BABA', 'MU']:
        print(f"🔍 分析 {symbol}...")
        metrics = calculate_metrics(df, symbol)
        if not metrics:
            print("  ❌ 无数据")
            continue
        
        result = analyze_stock_safe(symbol, metrics)
        
        print(f"  ✅ 评级: {result.investment_rating} (置信度: {result.confidence_score}/10)")
        print(f"  💰 目标价: 1月${result.target_price_1m}, 3月${result.target_price_3m}")
        
        if result.key_reasons:
            print(f"  📋 理由: {result.key_reasons[:2]}")
        if result.risk_factors:
            print(f"  ⚠️  风险: {result.risk_factors[:2]}")
        
        results.append({
            'symbol': symbol,
            'metrics': metrics,
            'analysis': result.model_dump()
        })
        print()
    
    # 保存报告
    report_file = f"kimi_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 报告已保存: {report_file}")
    
    # 打印汇总
    print("\n" + "="*70)
    print("📊 投资评级汇总")
    print("="*70)
    for r in results:
        a = r['analysis']
        m = r['metrics']
        print(f"{r['symbol']}: {a['investment_rating']:12} | 当前${m['current_price']} | 目标${a['target_price_1m']}")

if __name__ == "__main__":
    main()