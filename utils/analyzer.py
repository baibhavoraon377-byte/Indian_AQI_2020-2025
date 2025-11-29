import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import config

class AQIAnalyzer:
    def __init__(self):
        self.periods = config.ANALYSIS_PERIODS
    
    def analyze_trends(self, df):
        """Analyze overall trends 2020-2025"""
        yearly_trend = df.groupby('Year').agg({
            'AQI': ['mean', 'median', 'std', 'min', 'max'],
            'AQI_Category': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        return yearly_trend
    
    def compare_periods(self, df):
        """Compare different historical periods"""
        period_data = []
        for period_name, (start_date, end_date) in self.periods.items():
            period_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            if len(period_df) > 0:
                period_data.append({
                    'Period': period_name,
                    'Avg_AQI': period_df['AQI'].mean(),
                    'Median_AQI': period_df['AQI'].median(),
                    'Std_AQI': period_df['AQI'].std(),
                    'Days_Analyzed': len(period_df),
                    'Worst_State': period_df.groupby('State')['AQI'].mean().idxmax(),
                    'Best_State': period_df.groupby('State')['AQI'].mean().idxmin()
                })
        
        return pd.DataFrame(period_data)
    
    def identify_critical_states(self, df, year=None):
        """Identify states with critical AQI levels"""
        if year:
            df = df[df['Year'] == year]
        
        state_stats = df.groupby('State').agg({
            'AQI': ['mean', 'max', 'min', 'std'],
            'AQI_Category': lambda x: (x == 'Severe').sum()
        }).round(2)
        
        state_stats.columns = ['Avg_AQI', 'Max_AQI', 'Min_AQI', 'Std_AQI', 'Severe_Days']
        state_stats['Trend_2020_2025'] = self._calculate_trend(df)
        
        return state_stats.sort_values('Avg_AQI', ascending=False)
    
    def _calculate_trend(self, df):
        """Calculate AQI trend from 2020 to 2025"""
        trends = {}
        for state in df['State'].unique():
            state_data = df[df['State'] == state]
            yearly_avg = state_data.groupby('Year')['AQI'].mean()
            if len(yearly_avg) > 1:
                slope, _, _, _, _ = stats.linregress(range(len(yearly_avg)), yearly_avg.values)
                trends[state] = 'Improving' if slope < -1 else 'Worsening' if slope > 1 else 'Stable'
            else:
                trends[state] = 'Insufficient Data'
        
        return trends
    
    def analyze_seasonal_patterns(self, df):
        """Analyze seasonal patterns across years"""
        seasonal_stats = df.groupby(['Year', 'Season']).agg({
            'AQI': ['mean', 'std', 'count']
        }).round(2)
        
        seasonal_stats.columns = ['Avg_AQI', 'Std_AQI', 'Data_Points']
        return seasonal_stats
    
    def generate_health_impact_report(self, df):
        """Generate health impact analysis"""
        health_impact = df.groupby(['Year', 'AQI_Category']).size().unstack(fill_value=0)
        health_impact['Total_Days'] = health_impact.sum(axis=1)
        
        # Calculate percentage of days in each category
        for category in config.AQI_CATEGORIES.keys():
            if category in health_impact.columns:
                health_impact[f'{category}_Pct'] = (health_impact[category] / health_impact['Total_Days'] * 100).round(2)
        
        return health_impact