import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
import config

class AQIVisualizer:
    def __init__(self):
        self.colors = config.AQI_CATEGORIES
    
    def plot_yearly_trends(self, df):
        """Plot yearly AQI trends for 2020-2025"""
        yearly_avg = df.groupby(['State', 'Year'])['AQI'].mean().reset_index()
        
        fig = px.line(
            yearly_avg,
            x='Year',
            y='AQI',
            color='State',
            title='Yearly AQI Trends (2020-2025) by State',
            labels={'AQI': 'Average AQI', 'Year': 'Year'}
        )
        
        fig.update_layout(height=500, showlegend=True)
        return fig
    
    def plot_seasonal_analysis(self, df):
        """Plot seasonal patterns across years"""
        seasonal_avg = df.groupby(['Year', 'Season'])['AQI'].mean().reset_index()
        
        fig = px.bar(
            seasonal_avg,
            x='Year',
            y='AQI',
            color='Season',
            barmode='group',
            title='Seasonal AQI Patterns (2020-2025)',
            labels={'AQI': 'Average AQI'}
        )
        
        fig.update_layout(height=500)
        return fig
    
    def plot_state_comparison(self, df):
        """Compare states across different years"""
        state_year_avg = df.groupby(['State', 'Year'])['AQI'].mean().unstack().T
        
        fig = px.imshow(
            state_year_avg,
            aspect='auto',
            color_continuous_scale='RdYlGn_r',
            title='State-wise AQI Heatmap (2020-2025) - Red: Worse, Green: Better'
        )
        
        fig.update_layout(height=600)
        return fig
    
    def plot_critical_periods(self, df):
        """Analyze critical pollution periods"""
        critical_data = df[df['Is_Diwali_Period'] | df['Is_Crop_Burning_Season'] | df['Is_COVID_Lockdown']]
        
        fig = px.box(
            critical_data,
            x='Year',
            y='AQI',
            color='State',
            title='AQI Distribution During Critical Periods (2020-2025)',
            points='all'
        )
        
        fig.update_layout(height=500, showlegend=False)
        return fig
    
    def plot_pollutant_breakdown(self, df, year):
        """Breakdown of pollutants for a specific year"""
        year_data = df[df['Year'] == year]
        
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        available_pollutants = [p for p in pollutants if p in year_data.columns]
        
        if len(available_pollutants) < 2:
            return None
        
        pollutant_avg = year_data.groupby('State')[available_pollutants].mean().reset_index()
        pollutant_avg_melted = pollutant_avg.melt(id_vars=['State'], value_vars=available_pollutants,
                                                 var_name='Pollutant', value_name='Concentration')
        
        fig = px.bar(
            pollutant_avg_melted,
            x='State',
            y='Concentration',
            color='Pollutant',
            barmode='group',
            title=f'Pollutant Concentration by State ({year})'
        )
        
        fig.update_layout(height=500, xaxis_tickangle=-45)
        return fig
    
    def plot_geographical_distribution(self, df, year):
        """Geographical distribution of AQI for a specific year"""
        state_coords = {
            'Delhi': (28.6139, 77.2090), 'Uttar Pradesh': (26.8467, 80.9462),
            'Maharashtra': (19.0760, 72.8777), 'West Bengal': (22.5726, 88.3639),
            'Karnataka': (12.9716, 77.5946), 'Tamil Nadu': (13.0827, 80.2707),
            'Rajasthan': (26.9124, 75.7873), 'Gujarat': (23.0225, 72.5714),
            'Punjab': (31.1471, 75.3412), 'Haryana': (29.0588, 76.0856),
            'Bihar': (25.5941, 85.1376), 'Madhya Pradesh': (22.9734, 78.6569),
            'Andhra Pradesh': (17.3850, 78.4867), 'Telangana': (17.3850, 78.4867),
            'Kerala': (8.5241, 76.9366), 'Odisha': (20.2961, 85.8245),
            'Jharkhand': (23.6102, 85.2799), 'Assam': (26.2006, 92.9376)
        }
        
        year_data = df[df['Year'] == year]
        state_avg = year_data.groupby('State')['AQI'].mean().reset_index()
        state_avg['lat'] = state_avg['State'].map({state: coords[0] for state, coords in state_coords.items()})
        state_avg['lon'] = state_avg['State'].map({state: coords[1] for state, coords in state_coords.items()})
        state_avg = state_avg.dropna()
        
        fig = px.scatter_mapbox(
            state_avg,
            lat="lat",
            lon="lon",
            size="AQI",
            color="AQI",
            hover_name="State",
            hover_data={"AQI": True},
            color_continuous_scale="reds",
            size_max=30,
            zoom=4,
            title=f"Geographical AQI Distribution ({year})"
        )
        
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}, height=500)
        
        return fig
    
    def plot_category_evolution(self, df):
        """Evolution of AQI categories over years"""
        category_yearly = df.groupby(['Year', 'AQI_Category']).size().reset_index(name='Count')
        category_yearly_pivot = category_yearly.pivot(index='Year', columns='AQI_Category', values='Count').fillna(0)
        
        fig = px.area(
            category_yearly_pivot,
            title='Evolution of AQI Categories (2020-2025)',
            labels={'value': 'Number of Days', 'Year': 'Year'}
        )
        
        fig.update_layout(height=500)
        return fig