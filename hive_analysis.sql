USE stock_db;
SET hive.cli.print.header=true;

-- 🏆 收益率排名（使用正确的列名 close_price）
SELECT 
    symbol AS stock_symbol,
    name AS stock_name,
    COUNT(*) as trading_days,
    ROUND(MIN(close_price), 2) as min_close_price,
    ROUND(MAX(close_price), 2) as max_close_price,
    ROUND((MAX(close_price) - MIN(close_price)) / MIN(close_price) * 100, 2) as total_return_pct
FROM us_stocks
GROUP BY symbol, name
ORDER BY total_return_pct DESC;

-- 📊 详细统计
SELECT 
    symbol AS stock_symbol,
    name AS stock_name,
    ROUND(AVG(close_price), 2) as avg_close_price,
    ROUND((MAX(close_price) - MIN(close_price)) / MIN(close_price) * 100, 2) as total_return_pct,
    ROUND(AVG(daily_return) * 100, 2) as avg_daily_return_pct,
    ROUND(MAX(daily_return) * 100, 2) as max_gain_pct,
    ROUND(MIN(daily_return) * 100, 2) as max_loss_pct
FROM us_stocks
GROUP BY symbol, name
ORDER BY total_return_pct DESC;
