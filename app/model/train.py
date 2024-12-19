import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

from model.Model import Model_compeny
from model.data_manage import Data_engineering

from config.setting import SettingsManager
import joblib
import os

from datetime import datetime, timedelta

setting_info = SettingsManager("Setting file path")

project = setting_info.get_project_config("project name")

models = Model_compeny()


class trainer:

    def __init__(self, project: dict) -> None:
        # Store project settings
        self.project = project
        self.name = project['name']
        self.model_type = project['model']['type']

        # Extract model parameters
        self.params: list = []
        for i in list(project['model'])[1:]:
            self.params.append(project['model'][i])

        self.time_period = project["time_period"]

        # Create directory to save models
        self.path = os.path.join('Path of the models directory',
                                 self.name)
        os.makedirs(self.path, exist_ok=True)

    def hybrid_train(self, data):
        # Train hybrid model (Prophet + XGBoost) and return predictions and models
        params_1 = self.params[0]
        params_2 = self.params[1]

        # Get model dictionary
        model_dic = models.hybrid_model(params_1, params_2)
        prophet_model = model_dic['prophet']
        xgboost_model = model_dic['xgboost']

        # Data preparation
        data_manager = Data_engineering(data)
        df = data_manager.prophet_data()

        # Fit Prophet model
        prophet_model.fit(df)
        forecast = prophet_model.predict(df)

        # Prepare data for XGBoost
        X, y, df_xgb = data_manager.prepare_xgb_data(forecast['yhat'])

        # Fit XGBoost model
        xgboost_model.fit(X, y)
        residual_pred = xgboost_model.predict(X)

        # Combine Prophet predictions and XGBoost residual corrections
        df_xgb['residual_pred'] = residual_pred
        df_xgb['count_pred'] = forecast['yhat']

        return {'df': df_xgb, 'models': [prophet_model, xgboost_model]}

    def save_model(self, model: list):
        # Save trained models
        os.makedirs(self.path, exist_ok=True)

        # Handle hybrid and single model saving
        if self.model_type == "hybrid":
            for i in model:
                path = os.path.join(self.path, f"{i.name}.pkl")
                joblib.dump(i, path)
            return "ok"
        path = os.path.join(self.path, f"{model[0].name}.pkl")
        joblib.dump(model[0], path)
        return 'ok'

    def load_model(self):
        # Load models
        models: list = []
        for file in os.listdir(self.path):
            path = os.path.join(self.path, file)
            model = joblib.load(path)
            models.append(model)

        return models

    def train_save(self, data):
        # Train and save models
        if self.model_type == 'hybrid':
            models = self.hybrid_train(data=data)['models']
            self.save_model(model=models)
            return 'ok'

    def generate_future_dates(self,
                              start: int = 0,
                              end: int = 2,
                              freq: str = '30min'):
        # Generate future dates for prediction
        today = datetime.now().date()
        start_date = today + timedelta(days=start)
        end_date = today + timedelta(days=end)
        future_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

        future_df = pd.DataFrame({'ds': future_dates})
        return future_df

    def prediction(self,
                   start_day: int = 0,
                   end_day: int = 1,
                   freq: int = project["time_period"]):
        freq = str(freq) + "min"
        # print(freq)
        # Generate future data
        data = self.generate_future_dates(start=start_day,
                                          end=end_day,
                                          freq=freq)
        # Prepare data for Prophet models
        df = Data_engineering().furture_prophet_data(
            df=data, features=self.params[0]['regressors'])

        # Hybrid model prediction
        if self.model_type == 'hybrid':
            model_1, model_2 = self.load_model()
            forecast = model_1.predict(df)
            y = forecast['yhat']
            df['y'] = y

            # Prepare data for XGBoost part
            X = Data_engineering().furture_xgb_data(df=data)

            residual_pred = model_2.predict(X)

            final_y = y + residual_pred
            df['count'] = final_y

        return df, df[["timestamp", "count"]]
