# Large-Scale Financial Data Pipeline

美股数据智能分析与预测系统 - 基于大数据技术栈的金融数据分析项目

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Hadoop](https://img.shields.io/badge/Hadoop-3.3+-orange.svg)
![Hive](https://img.shields.io/badge/Hive-3.1-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)

## 项目概述

本项目构建了一套完整的金融数据处理流水线，支持：
- 美股历史数据采集与存储
- 大数据分析与指标计算
- LSTM 神经网络价格预测
- AI 驱动的投资分析报告生成

## 技术架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Data       │───▶│  Hadoop     │───▶│  Hive       │
│  Collection │    │  HDFS       │    │  Warehouse  │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
             ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
             │  Technical  │         │    LSTM     │         │   Kimi      │
             │  Analysis   │         │  Prediction │         │  AI Agent   │
             └─────────────┘         └─────────────┘         └─────────────┘
```

## 目录结构

```
bigdata-project/
├── download_us_stocks.py      # 数据采集：Yahoo Finance API
├── hive_stock.sql             # Hive 数据仓库建表语句
├── docker-compose.yml         # 大数据环境编排
├── model.py                   # LSTM 神经网络模型
├── stock_prediction.py        # 股票价格预测脚本
├── stock_kimi_langchain.py    # Kimi AI 分析集成
├── advanced_visualization.py # 数据可视化
└── jars/                     # Hadoop/Hive JAR 依赖
```

## 快速开始

### 1. 环境要求

- Python 3.12+
- Docker & Docker Compose
- Java 11+ (for Hadoop/Hive)

### 2. 安装依赖

```bash
pip install pandas yfinance torch scikit-learn matplotlib sqlalchemy pyhive
```

### 3. 启动大数据环境

```bash
docker-compose up -d
```

### 4. 数据采集

```bash
python download_us_stocks.py
```

### 5. 运行预测

```bash
python stock_prediction.py
```

## 核心功能

### 数据采集模块
- 支持 NVDA、TSLA、AAPL、MSFT 等 15+ 支热门股票
- 自动获取历史 OHLCV 数据
- 异常值检测与数据清洗

### 大数据存储
- **HDFS**: 分布式文件系统存储原始数据
- **Hive**: 数据仓库，支持 SQL 查询分析
- **分区策略**: 按股票代码 + 年份分区，优化查询性能
- **Parquet**: 列式存储，提升分析效率

### 技术指标计算
- 移动平均线 (MA5, MA20, MA60)
- MACD 指标
- 波动率分析
- 夏普比率

### 机器学习预测
- LSTM 神经网络架构
- 滚动窗口特征工程
- 7 天价格趋势预测
- 模型持久化与加载

### AI 增强分析
- LangChain 框架集成
- Kimi 大语言模型
- 自动生成投资分析报告

## 技术栈

| 层级 | 技术 |
|------|------|
| 数据采集 | Python, Pandas, Yahoo Finance API |
| 大数据存储 | Hadoop HDFS, Hive |
| 数据处理 | PySpark, SQL |
| 机器学习 | PyTorch, scikit-learn |
| AI 集成 | LangChain, Kimi API |
| 可视化 | Matplotlib, Plotly |
| 容器化 | Docker, Docker Compose |

## 数据仓库设计

```sql
-- 股票数据表
CREATE TABLE stock_prices (
    symbol STRING,
    date DATE,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT
)
PARTITIONED BY (year STRING, month STRING)
STORED AS PARQUET;
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

---

**Author**: Yao Zou  
**GitHub**: https://github.com/YaoZou23
