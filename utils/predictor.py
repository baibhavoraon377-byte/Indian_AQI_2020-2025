import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import config

class AQIPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        self.performance_metrics = {}
    
    def prepare_features(self, df, state, city):
        """Prepare features for a specific state and city"""
        location_data = df[(df['State'] == state) & (df['City'] == city)].copy()
        
        if len(location_data) < 60:
            return None, None, None, "Not enough data for prediction (minimum 60 days required)"
        
        location_data = location_data.sort_values('Date')
        
        base_features = ['Year', 'Month', 'DayOfYear', 'Weekday', 'Season', 'Week', 'Quarter']
        lag_features = [f'AQI_lag_{i}' for i in [1, 2, 3, 7, 14] if f'AQI_lag_{i}' in location_data.columns]
        rolling_features = [f'AQI_rolling_mean_{i}' for i in [7, 14] if f'AQI_rolling_mean_{i}' in location_data.columns]
        yoy_features = [f'AQI_yoy_lag_{i}' for i in [1] if f'AQI_yoy_lag_{i}' in location_data.columns]
        
        pollutant_features = [col for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 
                                            'Temperature', 'Humidity', 'Wind_Speed'] 
                            if col in location_data.columns]
        
        special_features = ['Is_Diwali_Period', 'Is_Crop_Burning_Season', 'Is_COVID_Lockdown']
        
        self.feature_columns = (base_features + lag_features + rolling_features + 
                              yoy_features + pollutant_features + special_features)
        
        model_data = location_data[self.feature_columns + ['AQI', 'Date']].dropna()
        
        if len(model_data) < 30:
            return None, None, None, "Not enough complete data for prediction"
        
        X = model_data[self.feature_columns]
        y = model_data['AQI']
        dates = model_data['Date']
        
        return X, y, dates, None
    
    def train_model(self, X, y):
        """Train the prediction model with multiple algorithms"""
        try:
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Try multiple models
            models = {
                'XGBoost': xgb.XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.1, random_state=42),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = name
                
                self.performance_metrics[name] = {
                    'r2': r2,
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                }
            
            self.model = best_model
            self.model_name = best_model_name
            self.is_trained = True
            
            return self.performance_metrics[best_model_name]
            
        except Exception as e:
            return None
    
    def predict_next_week(self, df, state, city, days=7):
        """Predict AQI for the next week"""
        if not self.is_trained:
            return None, "Model not trained"
        
        location_data = df[(df['State'] == state) & (df['City'] == city)].copy()
        location_data = location_data.sort_values('Date')
        
        if len(location_data) == 0:
            return None, "No data available for the selected location"
        
        predictions = []
        last_date = location_data['Date'].max()
        
        for day in range(1, days + 1):
            pred_date = last_date + timedelta(days=day)
            
            # Create feature set for prediction
            pred_features = self._create_prediction_features(location_data, pred_date)
            
            if pred_features is None:
                predictions.append({
                    'Date': pred_date,
                    'State': state,
                    'City': city,
                    'Predicted_AQI': None,
                    'AQI_Category': 'Unknown',
                    'Confidence': 'Low'
                })
                continue
            
            try:
                aqi_pred = self.model.predict(pred_features)[0]
                aqi_pred = max(0, min(500, aqi_pred))
                
                predictions.append({
                    'Date': pred_date,
                    'State': state,
                    'City': city,
                    'Predicted_AQI': round(aqi_pred, 2),
                    'AQI_Category': self._categorize_aqi(aqi_pred),
                    'Confidence': self._get_confidence_level(aqi_pred)
                })
                
            except Exception as e:
                predictions.append({
                    'Date': pred_date,
                    'State': state,
                    'City': city,
                    'Predicted_AQI': None,
                    'AQI_Category': 'Unknown',
                    'Confidence': 'Low',
                    'Error': str(e)
                })
        
        return pd.DataFrame(predictions), None
    
    def _create_prediction_features(self, location_data, pred_date):
        """Create feature set for a specific prediction date"""
        latest_data = location_data.iloc[-1:].copy()
        
        if latest_data.empty:
            return None
        
        pred_features = latest_data[self.feature_columns].copy()
        
        # Update date-based features
        pred_features['Year'] = pred_date.year
        pred_features['Month'] = pred_date.month
        pred_features['DayOfYear'] = pred_date.dayofyear
        pred_features['Weekday'] = pred_date.weekday
        pred_features['Week'] = pred_date.isocalendar().week
        pred_features['Quarter'] = pred_date.quarter
        pred_features['Season'] = self._get_season(pred_date.month)
        
        # Update special period markers
        pred_features['Is_Diwali_Period'] = self._is_diwali_period(pred_date)
        pred_features['Is_Crop_Burning_Season'] = pred_date.month in [10, 11]
        pred_features['Is_COVID_Lockdown'] = False
        
        return pred_features
    
    def _get_season(self, month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'
    
    def _is_diwali_period(self, date):
        diwali_dates = {
            2020: '2020-11-14', 2021: '2021-11-04', 2022: '2022-10-24',
            2023: '2023-11-12', 2024: '2024-10-31', 2025: '2025-10-20'
        }
        
        if date.year in diwali_dates:
            diwali = pd.Timestamp(diwali_dates[date.year])
            return (date >= diwali - timedelta(days=7)) and (date <= diwali + timedelta(days=7))
        return False
    
    def _categorize_aqi(self, aqi):
        for category, ranges in config.AQI_CATEGORIES.items():
            if ranges['min'] <= aqi <= ranges['max']:
                return category
        return 'Severe'
    
    def _get_confidence_level(self, aqi):
        if aqi < 100:
            return 'High'
        elif aqi < 300:
            return 'Medium'
        else:
            return 'Low'
    
    def get_feature_importance(self):
        if not self.is_trained or self.model is None:
            return None
        
        importance_scores = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        return feature_importance