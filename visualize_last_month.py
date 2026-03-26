#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    data_dir = "data/stocks"
    if not os.path.isdir(data_dir):
        raise ValueError("未找到 data/stocks 目录，请先运行下载脚本生成数据。")

    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            path = os.path.join(data_dir, file)
            df = pd.read_csv(path)
            if df.empty:
                continue
            all_data.append(df)

    if not all_data:
        raise ValueError("没有可用的股票数据文件，请先运行 download_us_stocks.py。")

    combined = pd.concat(all_data, ignore_index=True)
    if "date" not in combined.columns or "close" not in combined.columns:
        raise ValueError("CSV 缺少 date 或 close 列，请检查数据格式。")

    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date", "close"]).copy()
    if combined.empty:
        raise ValueError("日期或收盘价列无有效数据，无法可视化。")

    end_date = combined["date"].max()
    start_date = end_date - pd.Timedelta(days=30)
    last_month = combined[(combined["date"] >= start_date) & (combined["date"] <= end_date)].copy()
    if last_month.empty:
        raise ValueError("最近一个月没有可用数据。")

    plt.figure(figsize=(12, 7))
    for symbol in sorted(last_month["symbol"].dropna().unique()):
        stock_df = last_month[last_month["symbol"] == symbol].sort_values("date")
        if stock_df.empty:
            continue
        plt.plot(stock_df["date"], stock_df["close"], marker="o", linewidth=2, markersize=4, label=symbol)

    plt.title(f"Stock Close Price - Last Month ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    output = "last_month_prices.png"
    plt.savefig(output, dpi=160, bbox_inches="tight")
    print(f"✅ 图表已保存: {output}")


if __name__ == "__main__":
    main()
