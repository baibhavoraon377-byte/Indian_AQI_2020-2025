import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config

class DataLoader:
    def __init__(self):
        self.states = config.INDIAN_STATES
        self.aqi_categories = config.AQI_CATEGORIES
    
    def load_data(self, uploaded_file):
        """Load and validate uploaded CSV data for 2020-2025"""
        try:
            df = pd.read_csv(uploaded_file)
            
            required_columns = ['Date', 'State', 'City', 'AQI']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[(df['Date'].dt.year >= 2020) & (df['Date'].dt.year <= 2025)]
            
            if len(df) == 0:
                raise ValueError("No data found for the period 2020-2025")
            
            df = self._add_aqi_categories(df)
            df = self._engineer_features(df)
            df = self._add_period_markers(df)
            
            return df, None
            
        except Exception as e:
            return None, str(e)
    
    def _add_aqi_categories(self, df):
        """Add Indian AQI categories with health impacts"""
        def categorize_aqi(aqi):
            for category, ranges in self.aqi_categories.items():
                if ranges['min'] <= aqi <= ranges['max']:
                    return category
            return 'Severe'
        
        def get_health_impact(category):
            return self.aqi_categories.get(category, {}).get('health_impact', 'Unknown')
        
        df['AQI_Category'] = df['AQI'].apply(categorize_aqi)
        df['Health_Impact'] = df['AQI_Category'].apply(get_health_impact)
        df['AQI_Color'] = df['AQI_Category'].apply(lambda x: self.aqi_categories.get(x, {}).get('color', '#000000'))
        
        return df
    
    def _engineer_features(self, df):
        """Create comprehensive features for analysis"""
        df = df.sort_values(['State', 'City', 'Date'])
        
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Weekday'] = df['Date'].dt.weekday
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        df['Season'] = df['Month'].apply(self._get_season)
        
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'AQI_lag_{lag}'] = df.groupby(['State', 'City'])['AQI'].shift(lag)
        
        for window in [7, 14, 30]:
            df[f'AQI_rolling_mean_{window}'] = df.groupby(['State', 'City'])['AQI'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'AQI_rolling_std_{window}'] = df.groupby(['State', 'City'])['AQI'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        df['DayOfYear'] = df['Date'].dt.dayofyear
        for lag_year in [1, 2]:
            df[f'AQI_yoy_lag_{lag_year}'] = df.groupby(['State', 'City', 'DayOfYear'])['AQI'].shift(lag_year)
        
        if 'PM2.5' in df.columns and 'PM10' in df.columns:
            df['PM25_PM10_Ratio'] = df['PM2.5'] / df['PM10']
        
        return df
    
    def _get_season(self, month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'
    
    def _add_period_markers(self, df):
        df['Is_COVID_Lockdown'] = (df['Date'] >= '2020-03-25') & (df['Date'] <= '2020-06-30')
        
        diwali_dates = {
            '2020': '2020-11-14', '2021': '2021-11-04', '2022': '2022-10-24',
            '2023': '2023-11-12', '2024': '2024-10-31', '2025': '2025-10-20'
        }
        
        df['Is_Diwali_Period'] = False
        for year, diwali_date in diwali_dates.items():
            diwali = pd.Timestamp(diwali_date)
            df.loc[(df['Date'] >= diwali - timedelta(days=7)) & 
                  (df['Date'] <= diwali + timedelta(days=7)), 'Is_Diwali_Period'] = True
        
        df['Is_Crop_Burning_Season'] = df['Month'].isin([10, 11])
        
        return df
    
    def create_sample_2020_2025_data(self):
        """Create realistic sample data for 2020-2025"""
        dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')
        data = []
        
        for date in dates:
            for state, cities in self.states.items():
                for city in cities[:2]:  # Limit to 2 cities per state for sample
                    base_aqi = self._get_base_aqi(state, date)
                    aqi = self._apply_effects(base_aqi, state, date)
                    
                    row = {
                        'Date': date, 'State': state, 'City': city, 'AQI': aqi,
                        'PM2.5': aqi * 0.7 + np.random.normal(0, 15),
                        'PM10': aqi * 0.9 + np.random.normal(0, 20),
                        'NO2': aqi * 0.4 + np.random.normal(0, 10),
                        'SO2': aqi * 0.3 + np.random.normal(0, 8),
                        'CO': aqi * 0.15 + np.random.normal(0, 5),
                        'O3': aqi * 0.5 + np.random.normal(0, 12),
                        'Temperature': 25 + 15 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.normal(0, 5),
                        'Humidity': 60 + 30 * np.sin(2 * np.pi * date.dayofyear / 365 + np.pi) + np.random.normal(0, 15),
                        'Wind_Speed': 5 + 10 * np.random.random()
                    }
                    data.append(row)
        
        return pd.DataFrame(data)
    
    def _get_base_aqi(self, state, date):
        base_aqis = {
            'Delhi': 280, 'Uttar Pradesh': 220, 'Punjab': 200, 'Haryana': 250,
            'Rajasthan': 180, 'Gujarat': 170, 'Maharashtra': 140, 'West Bengal': 150,
            'Karnataka': 110, 'Tamil Nadu': 100, 'Bihar': 190, 'Madhya Pradesh': 160,
            'Andhra Pradesh': 130, 'Telangana': 120, 'Kerala': 90, 'Odisha': 110,
            'Jharkhand': 140, 'Assam': 80
        }
        
        base = base_aqis.get(state, 150)
        
        # COVID lockdown effect
        if '2020-03-25' <= date.strftime('%Y-%m-%d') <= '2020-06-30':
            base *= 0.6
        
        return base
    
    def _apply_effects(self, base_aqi, state, date):
        seasonal_effect = 50 * np.sin(2 * np.pi * (date.dayofyear - 300) / 365)
        
        diwali_effect = 0
        diwali_dates = ['2020-11-14', '2021-11-04', '2022-10-24', '2023-11-12', '2024-10-31', '2025-10-20']
        for diwali_date in diwali_dates:
            diwali = pd.Timestamp(diwali_date)
            days_from_diwali = abs((date - diwali).days)
            if days_from_diwali <= 10:
                diwali_effect = 150 * np.exp(-days_from_diwali / 3)
        
        crop_effect = 0
        if state in ['Delhi', 'Uttar Pradesh', 'Punjab', 'Haryana'] and date.month in [10, 11]:
            crop_effect = 100 * (1 - abs(date.month - 10.5) / 1.5)
        
        weekend_effect = -20 if date.weekday() >= 5 else 0
        
        noise = np.random.normal(0, 25)
        
        return max(30, base_aqi + seasonal_effect + diwali_effect + crop_effect + weekend_effect + noise)