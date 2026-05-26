import logging
import os

import joblib
import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from server.forecasting.csv_reader import read_retail_csv
from server.forecasting.record_cleaner import clean_retail_records
from server.forecasting.feature_engineer import engineer_sales_features

logger = logging.getLogger(__name__)

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

RF_MODEL_SAVE_PATH: str = os.path.join(_project_root, "model_vault", "rf_demand_model.pkl")
PIPELINE_SAVE_PATH: str = os.path.join(_project_root, "model_vault", "encoding_pipeline.pkl")


def run_rf_training(
    csv_path: str,
    holdout_ratio: float,
    num_trees: int,
    tree_depth: int,
    random_seed: int,
) -> dict:
    """Train a RandomForestRegressor on retail sales data and save the model."""
    os.makedirs(os.path.dirname(RF_MODEL_SAVE_PATH), exist_ok=True)

    raw_dataframe = read_retail_csv(csv_path)
    cleaned_dataframe = clean_retail_records(raw_dataframe)

    feature_matrix, target_column, feature_name_list, encoder_map, enriched_dataframe = engineer_sales_features(
        cleaned_dataframe
    )

    upper_bound = numpy.percentile(target_column, 99)
    valid_mask = target_column <= upper_bound
    feature_matrix = feature_matrix[valid_mask]
    target_column = target_column[valid_mask]
    logger.info("After 99th percentile filter: %d rows remain.", len(feature_matrix))

    split_index = int(len(feature_matrix) * (1 - holdout_ratio))
    training_features = feature_matrix.iloc[:split_index]
    training_targets = target_column.iloc[:split_index]
    holdout_features = feature_matrix.iloc[split_index:]
    holdout_targets = target_column.iloc[split_index:]

    best_model = RandomForestRegressor(
        n_estimators=num_trees,
        max_depth=tree_depth if tree_depth > 0 else None,
        max_features=0.5,
        bootstrap=True,
        n_jobs=-1,
        random_state=random_seed,
    )
    best_model.fit(training_features, training_targets)

    holdout_predictions = best_model.predict(holdout_features)
    mae_score = mean_absolute_error(holdout_targets, holdout_predictions)
    rmse_score = float(numpy.sqrt(mean_squared_error(holdout_targets, holdout_predictions)))
    r2_score_value = r2_score(holdout_targets, holdout_predictions)

    joblib.dump(best_model, RF_MODEL_SAVE_PATH)
    joblib.dump({"encoder_map": encoder_map, "feature_names": feature_name_list}, PIPELINE_SAVE_PATH)

    # Save metrics for Power BI dashboard
    import json
    metrics_path = os.path.join(os.path.dirname(RF_MODEL_SAVE_PATH), "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "mae": round(mae_score, 4),
            "rmse": round(rmse_score, 4),
            "r2": round(r2_score_value, 4),
            "num_trees": num_trees,
            "holdout_ratio": holdout_ratio,
            "feature_count": len(feature_name_list),
            "training_rows": len(training_features),
            "holdout_rows": len(holdout_features),
        }, f, indent=2)

    logger.info("Model saved to %s — MAE=%.4f, RMSE=%.4f, R2=%.4f", RF_MODEL_SAVE_PATH, mae_score, rmse_score, r2_score_value)

    return {
        "best_model": best_model,
        "mae": mae_score,
        "rmse": rmse_score,
        "r2": r2_score_value,
        "all_models": [best_model],
        "trained_models": [best_model],
    }
