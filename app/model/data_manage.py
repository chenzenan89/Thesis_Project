import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import savgol_filter


class Data_engineering:

    def __init__(self, data: dict[str, any] = None) -> None:
        # If data is provided, convert and process
        if data:
            self.data = pd.DataFrame(data)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data.set_index('timestamp', inplace=True)
            self.data_resampled = self.data.resample("30min").agg({
                'name':
                "first",
                "count":
                "mean"
            })
            self.data_resampled["count"] = self.data_resampled["count"].ffill()
            self.data_resampled = self.outlier(self.data_resampled)
            self.data_resampled = self.data_resampled.reset_index()
            self.data = self.data.reset_index()
        # Map strings to callbacks
        self.func = {
            "hour_scaled": self.add_hour_scale,
            "month_scaled": self.add_month_scaled,
            "is_weekend": self.add_is_weekend,
            "is_open": self.add_is_open
        }

    def outlier(self,
                df: pd.DataFrame,
                window: int = 10,
                k: int = 2) -> pd.DataFrame:
        # Calculate rolling mean and std
        df['rolling_mean'] = df['count'].rolling(window=window,
                                                 center=True).mean()
        df['rolling_std'] = df['count'].rolling(window=window,
                                                center=True).std()

        # Calculate upper and lower bounds for outliers
        df['upper_bound'] = df['rolling_mean'] + k * df['rolling_std']
        df['lower_bound'] = df['rolling_mean'] - k * df['rolling_std']

        # Flag outliers
        df['is_outlier'] = (df['count'] > df['upper_bound']) | (
            df['count'] < df['lower_bound'])

        # Drop temporary columns
        df = df.drop(columns=[
            'rolling_mean', 'rolling_std', 'upper_bound', 'lower_bound'
        ])
        return df

    def add_cyclic_and_discrete_features(self,
                                         df: pd.DataFrame) -> pd.DataFrame:
        # Add discrete time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['minute'] = df['timestamp'].dt.minute

        # Add cyclic representations of time
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        return df

    def smoothed(self, df: pd.DataFrame, item: str = 'count') -> pd.DataFrame:
        # Apply Savitzky-Golay filter for smoothing
        df[f"smoothed_{item}"] = savgol_filter(df[item],
                                               window_length=5,
                                               polyorder=2)
        return df

    def add_is_weekend(self, df: pd.DataFrame) -> pd.DataFrame:
        # Mark weekends (Saturday=5, Sunday=6) as 1, else 0
        df['is_weekend'] = df['timestamp'].dt.weekday.isin([5, 6]).astype(int)
        return df

    def add_hour_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        # Extract hour and create scaled hour feature
        df["hour"] = df["timestamp"].dt.hour
        df["hour_scaled"] = df["hour"] / 23
        return df

    def add_month_scaled(self, df: pd.DataFrame) -> pd.DataFrame:
        # Extract month and create scaled month feature
        df["month"] = df["timestamp"].dt.month
        df["month_scaled"] = df['month'] / 12.0
        return df

    def add_is_open(self,
                    df: pd.DataFrame,
                    open: int = 6,
                    close: int = 21) -> pd.DataFrame:
        # Mark as open (1) if within given hours, else 0
        df['is_open'] = ((df['timestamp'].dt.hour >= open) &
                         (df['timestamp'].dt.hour < close)).astype(int)
        return df

    def convert_column(self, df: pd.DataFrame, from_col: str,
                       to_col: str) -> pd.DataFrame:
        # Convert a column to datetime and rename
        if from_col in df.columns:
            df[to_col] = pd.to_datetime(df[from_col])
            df.drop(columns=[from_col], inplace=True)
        return df

    def prophet_data(self, include_features: bool = True) -> pd.DataFrame:
        # Create a copy of the resampled data
        df: pd.DataFrame = self.data_resampled.copy()

        # Weekend indicator
        df = self.add_is_weekend(df=df)

        # Scaled features
        df = self.add_hour_scale(df=df)
        df = self.add_month_scaled(df=df)

        # Handle outliers
        df = self.outlier(df)
        df['count'] = df['count'].where(~df['is_outlier'], np.nan)
        df['count'] = df['count'].ffill()

        # Apply smoothing
        df = self.smoothed(df=df)

        # Open/closed indicator
        df = self.add_is_open(df=df)

        # Rename the columns
        df = df.rename(columns={"timestamp": "ds", "count": "y"})

        # Include or exclude features as needed
        if include_features:
            return df[[
                'ds',
                'y',
                'hour_scaled',
                'month_scaled',
                "is_open",
                'is_weekend',
            ]]
        else:

            return df[['ds', 'y']]

    def prepare_xgb_data(
        self,
        baseline_predictions=None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
            Prepare data for training XGBRegressor, with residuals as target.
            Args:
                baseline_predictions (pd.Series or None): Baseline predictions for calculating residuals.
                                                        If None, a rolling mean is used as baseline.
            Returns:
                tuple: (X, y, df)
                X: Feature matrix (pd.DataFrame)
                y: Target residuals (pd.DataFrame)
                df: Processed DataFrame with features and residuals
        """
        df: pd.DataFrame = self.data_resampled.copy()
        df = self.outlier(df)
        df = self.add_cyclic_and_discrete_features(df)

        # If no baseline is provided, use rolling mean as baseline
        if baseline_predictions is None:
            baseline_predictions = df['count'].rolling(window=5,
                                                       min_periods=1).mean()
        df['baseline'] = baseline_predictions

        # Compute residuals
        df['residual'] = df['count'] - df['baseline']

        # Define features and residual columns
        features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'minute_sin',
            'minute_cos'
        ]
        residuals = ['residual']

        # Drop NaN values from features and residuals
        X = df[features].dropna()
        y = df[residuals].dropna()

        return X, y, df

    # Testing tools to facilitate development
    def visualization(df=None, t=48, a: list = []):
        plt.figure(figsize=(10, 6))
        if a:
            for i in a:
                plt.plot(i[:t],
                         label=str(i.name),
                         color=(random.random(), random.random(),
                                random.random()))
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid()
        plt.legend()
        plt.show()

    def furture_prophet_data(self, df: pd.DataFrame,
                             features: list[str]) -> pd.DataFrame:
        # Convert 'ds' to 'timestamp'
        df = self.convert_column(df=df, from_col='ds', to_col='timestamp')

        # Apply functions based on given features
        for feature in features:
            if feature in self.func.keys():
                fun = self.func.get(feature)
                df = fun(df)

        # Convert 'timestamp' back to 'ds'
        df = self.convert_column(df=df, from_col='timestamp', to_col='ds')
        return df

    def furture_xgb_data(self, df):
        # Convert 'ds' to 'timestamp'
        df = self.convert_column(df=df, from_col='ds', to_col='timestamp')

        # Add cyclic and discrete features
        df = self.add_cyclic_and_discrete_features(df=df)

        # Define the feature set
        features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'minute_sin',
            'minute_cos'
        ]

        # Return the feature matrix without NaNs
        X = df[features].dropna()

        return X

    def calculate_color(self,
                        value: float,
                        min_value: int = 0,
                        max_value: int = 100):
        # Map a value between min_value and max_value to a RGB gradient from green to red
        value = int(value)
        ratio = (value - min_value) / (max_value - min_value)
        red = int(255 * ratio)
        green = int(255 * (1 - ratio))
        return f"rgb({red},{green},0)"

    def dynamic_display(self, df: pd.DataFrame, startHour: int, endHour: int,
                        min: int, max: int) -> dict:

        # Filter data by hour range
        filtered_data = df[(df['timestamp'].dt.hour >= startHour)
                           & (df['timestamp'].dt.hour <= endHour)].copy()

        # Convert timestamp to string
        filtered_data['timestamp'] = filtered_data['timestamp'].dt.strftime(
            '%Y-%m-%d %H:%M')
        data_json = filtered_data.to_json(orient="records")

        # Preprocess count data
        filtered_data['real_count'] = filtered_data['count'].abs().astype(int)
        filtered_data["colors"] = filtered_data["real_count"].apply(
            lambda x: self.calculate_color(x, min, max))
        filtered_data['percentage'] = ((filtered_data['real_count'] - min) /
                                       (max - min)) * 100
        filtered_data['percentage'] = filtered_data['percentage'].round(2)

        # Extract date and time
        filtered_data['data'] = filtered_data['timestamp'].str.split(
            " ").str[0]
        filtered_data['time'] = filtered_data['timestamp'].str.split(
            " ").str[1]
        # Pivot table to get time as rows and date as columns
        table = filtered_data.pivot(index="time",
                                    columns="data",
                                    values="percentage")

        row_headers = table.index.tolist()
        col_headers = table.columns.tolist()
        values = table.values.tolist()

        # Generate a corresponding color table
        colors = table.map(
            lambda x: self.calculate_color(x, min, max)).values.tolist()

        return {
            'df': filtered_data,
            'json': data_json,
            'row_headers': row_headers,
            'col_headers': col_headers,
            'values': values,
            'colors': colors
        }
