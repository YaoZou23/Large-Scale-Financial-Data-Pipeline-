-- 启用动态分区
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;

-- 创建数据库
CREATE DATABASE IF NOT EXISTS stock_db;
USE stock_db;

-- 删除旧表（如果存在）
DROP TABLE IF EXISTS us_stocks;
DROP TABLE IF EXISTS us_stocks_temp;

-- 主表：分区表，Parquet列式存储
CREATE TABLE us_stocks (
    trade_date STRING,
    name STRING,
    open_price DOUBLE,
    high_price DOUBLE,
    low_price DOUBLE,
    close_price DOUBLE,
    volume BIGINT,
    ma5 DOUBLE,
    ma10 DOUBLE,
    ma20 DOUBLE,
    ma60 DOUBLE,
    macd DOUBLE,
    macd_signal DOUBLE,
    macd_hist DOUBLE,
    daily_return DOUBLE,
    price_change DOUBLE,
    volatility DOUBLE,
    volume_ma5 DOUBLE,
    volume_ratio DOUBLE,
    daily_range DOUBLE,
    daily_range_pct DOUBLE
)
PARTITIONED BY (symbol STRING, year INT)
STORED AS PARQUET;

-- 临时外部表：指向HDFS原始数据
CREATE EXTERNAL TABLE us_stocks_temp (
    trade_date STRING,
    symbol STRING,
    name STRING,
    open_price DOUBLE,
    high_price DOUBLE,
    low_price DOUBLE,
    close_price DOUBLE,
    volume BIGINT,
    ma5 DOUBLE,
    ma10 DOUBLE,
    ma20 DOUBLE,
    ma60 DOUBLE,
    macd DOUBLE,
    macd_signal DOUBLE,
    macd_hist DOUBLE,
    daily_return DOUBLE,
    price_change DOUBLE,
    volatility DOUBLE,
    volume_ma5 DOUBLE,
    volume_ratio DOUBLE,
    daily_range DOUBLE,
    daily_range_pct DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/stock_data/us/'
TBLPROPERTIES ("skip.header.line.count"="1");

-- 动态分区插入数据（分区字段必须在SELECT最后，且顺序一致）
INSERT OVERWRITE TABLE us_stocks PARTITION(symbol, year)
SELECT 
    trade_date,
    name,
    open_price, 
    high_price, 
    low_price, 
    close_price, 
    volume,
    ma5, 
    ma10, 
    ma20, 
    ma60,
    macd, 
    macd_signal, 
    macd_hist,
    daily_return, 
    price_change, 
    volatility,
    volume_ma5, 
    volume_ratio, 
    daily_range, 
    daily_range_pct,
    -- 分区字段必须是最后两个
    symbol,
    CAST(SUBSTRING(trade_date, 1, 4) AS INT) as year
FROM us_stocks_temp;

-- 收集统计信息（启用自动统计信息收集）
SET hive.stats.autogather=true;

-- 查看结果
SHOW PARTITIONS us_stocks;
SELECT COUNT(*) as total_rows FROM us_stocks;