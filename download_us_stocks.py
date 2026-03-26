#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import os
import subprocess

os.makedirs('data/stocks', exist_ok=True)

def download_and_process(symbol, name):
    """下载美股数据 + Pandas技术指标计算"""
    print(f"\n{'='*50}")
    print(f"处理 {name} ({symbol})")
    print(f"{'='*50}")
    
    # 1. 下载数据
    ticker = yf.Ticker(symbol)
    # 60日均线需要至少 60+ 个交易日，使用 1 年窗口保证指标可计算
    df = ticker.history(period="2y", interval="1d")

    if df.empty:
        raise ValueError(f"未获取到 {symbol} 数据")
    
    # 重置索引，date变为列
    df = df.reset_index()
    
    # 2. Pandas 数据处理
    # 标准化列名（小写）
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # 日期格式
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # 计算技术指标
    # 移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean().round(2)
    df['ma10'] = df['close'].rolling(window=10).mean().round(2)
    df['ma20'] = df['close'].rolling(window=20).mean().round(2)
    df['ma60'] = df['close'].rolling(window=60).mean().round(2)
    
    # 指数移动平均
    df['ema12'] = df['close'].ewm(span=12).mean().round(2)
    df['ema26'] = df['close'].ewm(span=26).mean().round(2)
    
    # MACD
    df['macd'] = (df['ema12'] - df['ema26']).round(2)
    df['macd_signal'] = df['macd'].ewm(span=9).mean().round(2)
    df['macd_hist'] = (df['macd'] - df['macd_signal']).round(2)
    
    # 涨跌幅
    df['daily_return'] = df['close'].pct_change().round(4)
    df['price_change'] = (df['close'] - df['open']).round(2)
    
    # 波动率（20日）
    df['volatility'] = df['daily_return'].rolling(window=20).std().round(4)
    
    # 成交量指标
    df['volume_ma5'] = df['volume'].rolling(window=5).mean().round(0)
    df['volume_ratio'] = (df['volume'] / df['volume_ma5']).round(2)
    
    # 价格区间
    df['daily_range'] = (df['high'] - df['low']).round(2)
    df['daily_range_pct'] = (df['daily_range'] / df['open'] * 100).round(2)
    
    # 股票信息
    df['symbol'] = symbol
    df['name'] = name
    
    # 选择最终列
    columns = ['date', 'symbol', 'name', 'open', 'high', 'low', 'close', 
               'volume', 'ma5', 'ma10', 'ma20', 'ma60',
               'macd', 'macd_signal', 'macd_hist',
               'daily_return', 'price_change', 'volatility',
               'volume_ma5', 'volume_ratio', 'daily_range', 'daily_range_pct']
    
    df = df[columns].dropna()
    if df.empty:
        raise ValueError(
            f"{symbol} 处理后无有效数据。请检查下载周期是否足够覆盖技术指标窗口（如 ma60）。"
        )
    
    # 3. 保存
    filename = f"data/stocks/{symbol}_processed.csv"
    df.to_csv(filename, index=False)
    
    # 打印统计
    print(f"✅ 数据条数: {len(df)}")
    print(f"📅 日期范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"💰 价格范围: ${df['close'].min():.2f} ~ ${df['close'].max():.2f}")
    print(f"📊 平均日涨跌: {df['daily_return'].mean()*100:.2f}%")
    print(f"🔥 最大单日涨幅: {df['daily_return'].max()*100:.2f}%")
    print(f"❄️ 最大单日跌幅: {df['daily_return'].min()*100:.2f}%")
    print(f"\n最近5天数据:")
    print(df.tail(5)[['date', 'close', 'ma5', 'macd', 'daily_return']].to_string())
    
    return filename

def upload_to_hdfs(local_file):
    """上传到HDFS"""
    filename = os.path.basename(local_file)
    
    try:
        # 复制到容器
        subprocess.run(
            ['docker', 'cp', local_file, f'namenode:/tmp/{filename}'],
            check=True,
            capture_output=True,
            text=True
        )

        # 上传到HDFS（-f 覆盖同名文件）
        subprocess.run(
            ['docker', 'exec', 'namenode', 'hdfs', 'dfs', '-put', '-f',
             f'/tmp/{filename}', '/stock_data/us/'],
            check=True,
            capture_output=True,
            text=True
        )
    finally:
        # 清理容器内临时文件（无论成功失败都执行）
        subprocess.run(
            ['docker', 'exec', 'namenode', 'rm', '-f', f'/tmp/{filename}'],
            capture_output=True,
            text=True
        )
    
    print(f"✅ 已上传 HDFS: /stock_data/us/{filename}")

if __name__ == "__main__":
    # 创建HDFS目录
    subprocess.run(['docker', 'exec', 'namenode', 'hdfs', 'dfs', 
                   '-mkdir', '-p', '/stock_data/us/'], 
                   check=True, capture_output=True)
    
    # 美股列表
    stocks = [
    # 科技板块
    ("NVDA", "NVIDIA"), ("AAPL", "Apple"), ("MSFT", "Microsoft"),
    ("GOOGL", "Google"), ("META", "Meta"), ("AMD", "AMD"),
    # 金融板块
    ("JPM", "JPMorgan"), ("BAC", "Bank of America"), ("GS", "Goldman"),
    # 消费板块
    ("AMZN", "Amazon"), ("TSLA", "Tesla"), ("WMT", "Walmart"),
    # 中概股
    ("BABA", "Alibaba"), ("PDD", "PDD"), ("JD", "JD"),
    ("NIO", "NIO"), ("LI", "Li Auto"), ("XPEV", "Xpeng"),
    ]

    
    uploaded_files = []
    
    for symbol, name in stocks:
        try:
            file = download_and_process(symbol, name)
            upload_to_hdfs(file)
            uploaded_files.append(file)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or "").strip()
            print(f"❌ {symbol} 失败: {e}")
            if err:
                print(f"   ↳ 错误详情: {err}")
        except Exception as e:
            print(f"❌ {symbol} 失败: {e}")
    
    # 最终报告
    print(f"\n{'='*50}")
    print("📊 处理完成报告")
    print(f"{'='*50}")
    print(f"成功处理: {len(uploaded_files)}/{len(stocks)} 只股票")
    
    print("\n📂 HDFS 文件列表:")
    result = subprocess.run(['docker', 'exec', 'namenode', 'hdfs', 'dfs', 
                            '-ls', '/stock_data/us/'], 
                           capture_output=True, text=True)
    print(result.stdout)
    
    if result.stderr:
        print("错误:", result.stderr)