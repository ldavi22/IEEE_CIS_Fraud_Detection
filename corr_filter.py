import numpy as np
import dagshub
import mlflow
def filter(x_train, y_train, threshold, x_valid=None):
    corr_matrix = np.abs(x_train.corr())
    cols = corr_matrix.columns.tolist()
    to_drop = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[i] in to_drop or cols[j] in to_drop:
                continue
            if corr_matrix.loc[cols[i], cols[j]] > threshold:
                corr_i = abs(x_train[cols[i]].corr(y_train))
                corr_j = abs(x_train[cols[j]].corr(y_train))
                if corr_i >= corr_j:
                    to_drop.add(cols[j])
                else:
                    to_drop.add(cols[i])

    x_train_filtered = x_train.drop(columns=list(to_drop))

    if x_valid is not None:
        x_valid_filtered = x_valid.drop(columns=list(to_drop))
        return x_train_filtered, x_valid_filtered

    return x_train_filtered


def logging(grid,x_train_filtered,x_valid_filtered,y_train,y_valid,run_name):
    import dagshub
    import mlflow
    from sklearn.metrics import roc_auc_score
    dagshub.init(repo_owner='ldavi22', repo_name='IEEE_CIS_Fraud_Detection', mlflow=True)

    result = grid.fit(x_train_filtered, y_train)
    best_model = result.best_estimator_
    y_valid_pred = best_model.predict_proba(x_valid_filtered)[:, 1]
    valid_auc = roc_auc_score(y_valid, y_valid_pred)
    mlflow.set_experiment("XGBoost_Tuning")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("preprocessing", "High NaN Dropping / Mean-Median Fill")
        mlflow.set_tag("data_split", "Time-Based Split")
        mlflow.set_tag("algorithm", "XGBoost")

        for key, val in result.best_params_.items():
            mlflow.log_param(key, val)

        mlflow.log_param("n_features", x_valid_filtered.shape[1])
        mlflow.log_param("validation_samples", x_valid_filtered.shape[0])

        mlflow.log_metric("train_auc", result.best_score_)
        mlflow.log_metric("valid_auc", valid_auc)