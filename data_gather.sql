CREATE OR REPLACE TABLE
  `u4cast-cr.iowa_liquor_sales.top_100_stores` AS
SELECT
  store_number,
  SUM(sale_dollars) AS sales
FROM
  `bigquery-public-data.iowa_liquor_sales.sales`
WHERE
  date >= '2018-01-01'
  AND date < '2022-01-01'
  AND store_number IN (
  SELECT
    store_number
  FROM (
    SELECT
      store_number,
      MIN(date) AS min_date,
      MAX(date) AS max_date
    FROM
      `bigquery-public-data.iowa_liquor_sales.sales`
    GROUP BY
      1 ) a
  WHERE
    min_date < '2018-01-01'AND max_date > '2022-03-01')
GROUP BY
  1
ORDER BY
  sales DESC
LIMIT
  100