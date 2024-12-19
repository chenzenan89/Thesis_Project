from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


class Model_compeny:

    def __init__(self):
        pass

    def prophet(self,
                changepoint_prior_scale=0.8,
                seasonality_mode='multiplicative',
                seasonality_prior_scale=15.0,
                regressors: list = None,
                seasonality=True,
                holidays=None):
        """
        Creates and configures a Prophet model.
        
        Args:
            c (float): Changepoint prior scale. Controls flexibility of trend changes.
            mode (str): Seasonality mode ('additive' or 'multiplicative').
            s (float): Seasonality prior scale. Controls flexibility of seasonality.
            regressors (list, optional): List of additional regressors to add.
            seasonality (bool): Whether to include custom seasonality components.
            holidays (pd.DataFrame, optional): DataFrame defining holiday effects.
        
        Returns:
            Prophet: Configured Prophet model.
        """
        model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_mode=seasonality_mode,
                        seasonality_prior_scale=seasonality_prior_scale,
                        holidays=holidays)

        if regressors:
            for features in regressors:
                model.add_regressor(features)
        if seasonality:
            model.add_seasonality(name='hourly', period=24, fourier_order=12)
            model.add_seasonality(name='half_hourly',
                                  period=1,
                                  fourier_order=12)

        return model

    def xgbregressor(self,
                     n_estimators=500,
                     learning_rate=0.1,
                     max_depth=7,
                     subsample=0.8,
                     colsample_bytree=0.8):
        """
        Creates and configures an XGBRegressor model.

        Args:
            n (int): Number of estimators (trees).
            lr (float): Learning rate.
            max_depth (int): Maximum depth of trees.
            subsample (float): Fraction of samples to use per tree.
            colsample_bytree (float): Fraction of features to use per tree.

        Returns:
            XGBRegressor: Configured XGBRegressor model.
        """

        model = XGBRegressor(n_estimators=n_estimators,
                             learning_rate=learning_rate,
                             max_depth=max_depth,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree)

        return model

    def hybrid_model(self, prophet_params, xgb_params):
        """
        Creates a hybrid model combining Prophet and XGBoost.
        
        Args:
            prophet_params (dict): Parameters for the Prophet model.
            xgb_params (dict): Parameters for the XGBoost model.
        
        Returns:
            dict: Contains both Prophet and XGBoost models.
        """
        prophet_model = self.prophet(**prophet_params)
        xgb_model = self.xgbregressor(**xgb_params)
        prophet_model.name = 'prophet'
        xgb_model.name = 'xgboost'
        return {"prophet": prophet_model, "xgboost": xgb_model}

    def create_model(self, model_type, params):
        """
        Dynamically creates a model instance based on the type and parameters.

        Args:
            model_type (str): The type of model ('prophet', 'xgbregressor', or 'hybrid').
            params (dict): Parameters for the model.

        Returns:
            object or dict: Configured model instance or dictionary for hybrid models.
        """
        if model_type == "prophet":
            return self.prophet(**params)
        elif model_type == "xgbregressor":
            return self.xgbregressor(**params)
        elif model_type == "hybrid":
            return self.hybrid_model(params["prophet_params"],
                                     params["xgb_params"])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def evaluate_model(self, y_true, y_pred):
        """
        Evaluate the model's performance using various metrics.

        Args:
            y_true (array-like): Ground truth target values.
            y_pred (array-like): Predicted target values.

        Returns:
            dict: Performance metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}
        return metrics
