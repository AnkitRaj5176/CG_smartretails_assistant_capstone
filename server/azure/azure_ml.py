"""
azure_ml.py
───────────
Azure Machine Learning integration.

Supports:
  - Registering trained models to Azure ML Model Registry
  - Submitting training jobs to Azure ML compute
  - Deploying models as Azure ML Online Endpoints
  - Fetching model metrics from Azure ML runs

Falls back to local model operations when Azure ML not configured.
"""

import logging
import os

from server.env_config import env_settings

logger = logging.getLogger(__name__)

_ml_client = None

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RF_MODEL_PATH = os.path.join(_project_root, "model_vault", "rf_demand_model.pkl")
_PIPELINE_PATH = os.path.join(_project_root, "model_vault", "encoding_pipeline.pkl")


def _get_ml_client():
    """Return cached Azure ML client or None."""
    global _ml_client
    if _ml_client is not None:
        return _ml_client
    if not env_settings.USE_AZURE_ML:
        return None
    try:
        from azure.ai.ml import MLClient                        # type: ignore
        from azure.identity import DefaultAzureCredential      # type: ignore
        _ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=env_settings.AZURE_ML_SUBSCRIPTION_ID,
            resource_group_name=env_settings.AZURE_ML_RESOURCE_GROUP,
            workspace_name=env_settings.AZURE_ML_WORKSPACE,
        )
        logger.info("Azure ML client initialised (workspace=%s).", env_settings.AZURE_ML_WORKSPACE)
        return _ml_client
    except ImportError:
        logger.warning("azure-ai-ml not installed.")
        return None
    except Exception as e:
        logger.warning("Azure ML client init failed: %s", e)
        return None


# ── Model Registration ─────────────────────────────────────────────────────────

def register_model(model_name: str = "retail-demand-rf", version: str = "1") -> dict:
    """
    Register the trained RandomForest model in Azure ML Model Registry.
    """
    client = _get_ml_client()
    if not client:
        return {
            "status": "local",
            "message": "Azure ML not configured — model saved locally in model_vault/",
            "local_path": _RF_MODEL_PATH,
        }

    if not os.path.exists(_RF_MODEL_PATH):
        return {"status": "error", "message": "Model not trained yet. Run /api/ml/train first."}

    try:
        from azure.ai.ml.entities import Model                  # type: ignore
        from azure.ai.ml.constants import AssetTypes            # type: ignore
        model = Model(
            path=_RF_MODEL_PATH,
            name=model_name,
            description="RandomForest demand forecasting model for Smart Retail Analytics Engine",
            type=AssetTypes.CUSTOM_MODEL,
        )
        registered = client.models.create_or_update(model)
        logger.info("Model registered in Azure ML: %s v%s", registered.name, registered.version)
        return {
            "status": "registered",
            "model_name": registered.name,
            "version": registered.version,
            "workspace": env_settings.AZURE_ML_WORKSPACE,
        }
    except Exception as e:
        logger.warning("Model registration failed: %s", e)
        return {"status": "error", "detail": str(e)}


# ── Training Job ───────────────────────────────────────────────────────────────

def submit_training_job(
    compute_name: str = "retail-compute",
    num_trees: int = 100,
    holdout_ratio: float = 0.2,
) -> dict:
    """
    Submit a training job to Azure ML compute cluster.
    Falls back to local training when Azure ML not configured.
    """
    client = _get_ml_client()
    if not client:
        # Run training locally
        csv_path = os.path.join(_project_root, "sales_store", "retail_records.csv")
        if not os.path.exists(csv_path):
            return {"status": "error", "message": "No data uploaded. Use /api/data/upload first."}
        from server.forecasting.rf_trainer import run_rf_training
        result = run_rf_training(
            csv_path=csv_path,
            holdout_ratio=holdout_ratio,
            num_trees=num_trees,
            tree_depth=0,
            random_seed=42,
        )
        return {
            "status": "completed_locally",
            "mae": round(result["mae"], 4),
            "rmse": round(result["rmse"], 4),
            "r2": round(result["r2"], 4),
            "compute": "local",
        }

    try:
        from azure.ai.ml import command                         # type: ignore
        from azure.ai.ml.entities import Environment            # type: ignore
        job = command(
            code=os.path.join(_project_root, "server"),
            command=(
                f"python -c \""
                f"from forecasting.rf_trainer import run_rf_training; "
                f"run_rf_training('data/retail_records.csv', {holdout_ratio}, {num_trees}, 0, 42)"
                f"\""
            ),
            environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
            compute=compute_name,
            display_name="retail-demand-training",
            description="Train RandomForest demand forecasting model",
        )
        submitted = client.jobs.create_or_update(job)
        logger.info("Azure ML training job submitted: %s", submitted.name)
        return {
            "status": "submitted",
            "job_name": submitted.name,
            "compute": compute_name,
            "workspace": env_settings.AZURE_ML_WORKSPACE,
        }
    except Exception as e:
        logger.warning("Azure ML job submission failed: %s", e)
        return {"status": "error", "detail": str(e)}


# ── Model Deployment ───────────────────────────────────────────────────────────

def get_deployment_status() -> dict:
    """Return Azure ML deployment status or local model status."""
    client = _get_ml_client()

    local_model_exists = os.path.exists(_RF_MODEL_PATH)
    local_pipeline_exists = os.path.exists(_PIPELINE_PATH)

    if not client:
        return {
            "deployment_type": "local",
            "model_ready": local_model_exists,
            "pipeline_ready": local_pipeline_exists,
            "inference_endpoint": "http://localhost:8000/api/ml/predict",
            "workspace": "N/A (Azure ML not configured)",
        }

    try:
        workspaces = list(client.workspaces.list())
        return {
            "deployment_type": "azure_ml",
            "workspace": env_settings.AZURE_ML_WORKSPACE,
            "model_ready": local_model_exists,
            "inference_endpoint": f"https://{env_settings.AZURE_WEBAPP_NAME}.azurewebsites.net/api/ml/predict",
            "azure_ml_connected": True,
        }
    except Exception as e:
        return {
            "deployment_type": "azure_ml",
            "azure_ml_connected": False,
            "error": str(e),
            "model_ready": local_model_exists,
        }
