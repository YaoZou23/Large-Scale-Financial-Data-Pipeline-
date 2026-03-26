USE stock_db;

-- 查看 NVDA 所有数据（按日期排序）
describe us_stocks;

SELECT 
    symbol,
    name,
    COUNT(*) as up_days,
    MIN(trade_date) as start_date,
    MAX(trade_date) as end_date,
    ROUND((MAX(close_price) - MIN(close_price)) / MIN(close_price) * 100, 2) as total_gain_pct
FROM us_stocks
WHERE daily_return > 0
    AND trade_date >= date_sub((SELECT MAX(trade_date) FROM us_stocks), 6)
GROUP BY symbol, name
HAVING COUNT(*) >= 5  -- 至少5天上涨
ORDER BY up_days DESC, total_gain_pct DESC;