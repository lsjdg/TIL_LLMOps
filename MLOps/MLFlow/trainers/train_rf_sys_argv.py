import mlflow
import numpy as np
import sys

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.sklearn.autolog()

data_diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data_diabetes.data, data_diabetes.target
)

if __name__ == "__main__":
    # n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    # max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # reg = RandomForestRegressor(n_estimators=n_estimators,
    #                                                       max_depth=max_depth,
    #                                                       random_state=42)
    # reg.fit(X_train, y_train)

    model_name = sys.argv[1]

    if model_name == "linear":
        from sklearn.linear_model import LinearRegression

        reg = LinearRegression()
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        n_estimators = int(sys.argv[2]) if len(sys.argv) > 1 else 100
        max_depth = int(sys.argv[3]) if len(sys.argv) > 2 else None

        reg = RandomForestRegressor()
    else:
        from sklearn.ensemble import GradientBoostingRegressor

        reg = GradientBoostingRegressor()

    reg.fit(X_train, y_train)


# python train_rf_sys_argv.py linear
# python train_rf_sys_argv.py linear
# python train_rf_sys_argv.py rf 100 10
# python train_rf_sys_argv.py aaaa
