INSERT INTO `proj.scale_forecasting.forecast_predictions`
  (run_id, ts_id, model_type, compute_engine, forecast_date,
   yhat, yhat_lower, yhat_upper, quantiles)
WITH base_pred AS (
  SELECT run_id, ts_id, model_type, forecast_date, yhat, yhat_lower, yhat_upper
  FROM `proj.scale_forecasting.forecast_predictions`
  WHERE run_id = @run_id
    AND model_type IN ('theta', 'sarimax', 'xgboost')
    AND model_type NOT IN (
      SELECT model_type FROM `proj.scale_forecasting.forecast_metadata`
      WHERE run_id = @run_id AND wape > 0.3
    )
)
SELECT run_id, ts_id, 'ensemble_mean' AS model_type,
       'ensemble' AS compute_engine, forecast_date,
       AVG(yhat) AS yhat, AVG(yhat_lower) AS yhat_lower, AVG(yhat_upper) AS yhat_upper,
       CAST(NULL AS STRING) AS quantiles
FROM base_pred
GROUP BY run_id, ts_id, forecast_date;

INSERT INTO `proj.scale_forecasting.forecast_predictions`
  (run_id, ts_id, model_type, compute_engine, forecast_date,
   yhat, yhat_lower, yhat_upper, quantiles)
WITH base_pred AS (
  SELECT run_id, ts_id, model_type, forecast_date, yhat, yhat_lower, yhat_upper
  FROM `proj.scale_forecasting.forecast_predictions`
  WHERE run_id = @run_id
    AND model_type IN ('theta', 'sarimax', 'xgboost')
    AND model_type NOT IN (
      SELECT model_type FROM `proj.scale_forecasting.forecast_metadata`
      WHERE run_id = @run_id AND wape > 0.3
    )
)
SELECT run_id, ts_id, 'ensemble_median' AS model_type,
       'ensemble' AS compute_engine, forecast_date,
       APPROX_QUANTILES(yhat, 2)[OFFSET(1)] AS yhat, APPROX_QUANTILES(yhat_lower, 2)[OFFSET(1)] AS yhat_lower, APPROX_QUANTILES(yhat_upper, 2)[OFFSET(1)] AS yhat_upper,
       CAST(NULL AS STRING) AS quantiles
FROM base_pred
GROUP BY run_id, ts_id, forecast_date;

INSERT INTO `proj.scale_forecasting.forecast_predictions`
  (run_id, ts_id, model_type, compute_engine, forecast_date,
   yhat, yhat_lower, yhat_upper, quantiles)
WITH base_pred AS (
  SELECT run_id, ts_id, model_type, forecast_date, yhat, yhat_lower, yhat_upper
  FROM `proj.scale_forecasting.forecast_predictions`
  WHERE run_id = @run_id
    AND model_type IN ('theta', 'sarimax', 'xgboost')
    AND model_type NOT IN (
      SELECT model_type FROM `proj.scale_forecasting.forecast_metadata`
      WHERE run_id = @run_id AND wape > 0.3
    )
),
model_weight AS (
  SELECT ts_id, model_type,
         SAFE_DIVIDE(1, NULLIF(AVG(wape), 0)) AS w
  FROM `proj.scale_forecasting.forecast_metadata`
  WHERE run_id = @run_id AND model_type IN ('theta', 'sarimax', 'xgboost')
  GROUP BY ts_id, model_type
)
SELECT p.run_id, p.ts_id, 'ensemble_inverse_error' AS model_type,
       'ensemble' AS compute_engine, p.forecast_date,
       SAFE_DIVIDE(SUM(p.yhat * w.w), SUM(w.w)) AS yhat,
       SAFE_DIVIDE(SUM(p.yhat_lower * w.w), SUM(w.w)) AS yhat_lower,
       SAFE_DIVIDE(SUM(p.yhat_upper * w.w), SUM(w.w)) AS yhat_upper,
       CAST(NULL AS STRING) AS quantiles
FROM base_pred p
JOIN model_weight w USING (ts_id, model_type)
WHERE w.w IS NOT NULL
GROUP BY p.run_id, p.ts_id, p.forecast_date;