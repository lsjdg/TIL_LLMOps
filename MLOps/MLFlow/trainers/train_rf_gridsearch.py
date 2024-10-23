import mlflow
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.sklearn.autolog()

data_diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data_diabetes.data, data_diabetes.target
)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# __name__ : 현재 실행한 환경 이름. python 명령어를 이용해서 실행하면 __main__이 된다
# __main__ : entry point
if __name__ == "__main__":
    # GridSearchCV에 대한 Logging은 autolog()를 이용해 자동으로 저장
    param_grid = {
        "n_estimators": [10, 20, 30, 40, 50, 100],
        "max_depth": [10, 15, 20],
        "max_features": [5, 6, 7, 8, 9],
    }

    grid_search = GridSearchCV(
        RandomForestRegressor(),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )

    grid_search.fit(X_train, y_train)

    # train set 에 대한 예측 기록 저장
    predictions = grid_search.best_estimator_.predict(X_train)
    # best model 의 훈련 성과가 저장될 객체
    signature = infer_signature(X_train, predictions)

    # Best Estimator에 대한 결과물을 따로 로깅하기

    # 최고의 성능을 냈던 하이퍼 파라미터를 따로 로깅
    mlflow.log_param("alpha", grid_search.best_params_["n_estimators"])
    mlflow.log_param("max_depth", grid_search.best_params_["max_depth"])
    mlflow.log_param("max_features", grid_search.best_params_["max_features"])

    # 최고의 성능을 냈던 메트릭을 따로 로깅
    rmse, r2, mae = eval_metrics(y_test, grid_search.best_estimator_.predict(X_test))

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(grid_search.best_estimator_, "model", signature=signature)
