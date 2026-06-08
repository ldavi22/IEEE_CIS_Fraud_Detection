import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature


def log_experiment(
        model,
        experiment_name,
        run_name,
        best_params,
        train_auc,
        y_valid,
        y_valid_pred,
        x_valid_filtered,
        save_model=False,
):
    """
    Simple wrapper to log experiment to MLflow.

    Args:
        model: Trained model object
        experiment_name: Name of the experiment (e.g., "XGBoost_Tuning")
        run_name: Name of this run (e.g., "XGBoost_v1")
        best_params: Dict of best hyperparameters
        train_auc: Training AUC score
        y_valid: Validation labels
        y_valid_pred: Predicted probabilities on validation set
        x_valid_filtered: Validation features
        save_model: Boolean, whether to save model artifact (default: False)
    """

    valid_auc = roc_auc_score(y_valid, y_valid_pred)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        for key, val in best_params.items():
            mlflow.log_param(key, val)

        # Log metrics
        mlflow.log_metric("train_auc", train_auc)
        mlflow.log_metric("valid_auc", valid_auc)

        # Save model if requested
        if save_model:
            signature = infer_signature(x_valid_filtered, y_valid_pred)
            mlflow.sklearn.log_model(
                model,
                "xgboost_model",
                signature=signature,
            )