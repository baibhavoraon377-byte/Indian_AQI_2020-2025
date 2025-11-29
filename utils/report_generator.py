import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import config
from datetime import datetime

class ReportGenerator:
    def __init__(self):
        self.aqi_categories = config.AQI_CATEGORIES
    
    def generate_comprehensive_report(self, df, analyzer):
        """Generate a comprehensive AQI analysis report"""
        report = {}
        
        # Executive Summary
        report['executive_summary'] = self._generate_executive_summary(df)
        
        # Key Metrics
        report['key_metrics'] = self._generate_key_metrics(df)
        
        # Trend Analysis
        report['trend_analysis'] = self._generate_trend_analysis(df)
        
        # Critical States Analysis
        report['critical_states'] = self._generate_critical_states_analysis(df, analyzer)
        
        # Seasonal Analysis
        report['seasonal_analysis'] = self._generate_seasonal_analysis(df)
        
        # Health Impact Assessment
        report['health_impact'] = self._generate_health_impact(df)
        
        # Recommendations
        report['recommendations'] = self._generate_recommendations(df, analyzer)
        
        return report
    
    def _generate_executive_summary(self, df):
        """Generate executive summary"""
        total_days = len(df['Date'].unique())
        states_covered = df['State'].nunique()
        cities_covered = df['City'].nunique()
        
        # Overall AQI statistics
        avg_aqi = df['AQI'].mean()
        max_aqi = df['AQI'].max()
        min_aqi = df['AQI'].min()
        
        # Critical days count
        severe_days = len(df[df['AQI_Category'] == 'Severe'])
        very_poor_days = len(df[df['AQI_Category'] == 'Very Poor'])
        critical_days = severe_days + very_poor_days
        
        summary = {
            'analysis_period': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
            'total_days_analyzed': total_days,
            'states_covered': states_covered,
            'cities_covered': cities_covered,
            'average_aqi': round(avg_aqi, 2),
            'maximum_aqi': max_aqi,
            'minimum_aqi': min_aqi,
            'critical_pollution_days': critical_days,
            'critical_days_percentage': round((critical_days / total_days) * 100, 2)
        }
        
        return summary
    
    def _generate_key_metrics(self, df):
        """Generate key performance metrics"""
        metrics = {}
        
        # Yearly metrics
        yearly_metrics = df.groupby('Year').agg({
            'AQI': ['mean', 'max', 'min', 'std'],
            'AQI_Category': lambda x: (x.isin(['Severe', 'Very Poor'])).sum()
        }).round(2)
        
        yearly_metrics.columns = ['Avg_AQI', 'Max_AQI', 'Min_AQI', 'Std_AQI', 'Critical_Days']
        metrics['yearly_metrics'] = yearly_metrics
        
        # State-wise rankings
        state_metrics = df.groupby('State').agg({
            'AQI': ['mean', 'max', 'std'],
            'AQI_Category': lambda x: (x.isin(['Severe', 'Very Poor'])).sum()
        }).round(2)
        
        state_metrics.columns = ['Avg_AQI', 'Max_AQI', 'Std_AQI', 'Critical_Days']
        metrics['state_rankings'] = state_metrics.sort_values('Avg_AQI', ascending=False)
        
        # Worst performing cities
        city_metrics = df.groupby(['State', 'City']).agg({
            'AQI': 'mean',
            'AQI_Category': lambda x: (x.isin(['Severe', 'Very Poor'])).sum()
        }).round(2)
        
        city_metrics.columns = ['Avg_AQI', 'Critical_Days']
        metrics['worst_cities'] = city_metrics.sort_values('Avg_AQI', ascending=False).head(10)
        
        return metrics
    
    def _generate_trend_analysis(self, df):
        """Generate trend analysis"""
        trends = {}
        
        # Yearly trend
        yearly_trend = df.groupby('Year')['AQI'].agg(['mean', 'std', 'count']).round(2)
        trends['yearly_trend'] = yearly_trend
        
        # Monthly trend
        monthly_trend = df.groupby(['Year', 'Month'])['AQI'].mean().unstack().round(2)
        trends['monthly_trend'] = monthly_trend
        
        # COVID impact analysis
        covid_period = df[df['Is_COVID_Lockdown']]
        pre_covid = df[df['Date'] < '2020-03-25']
        post_covid = df[df['Date'] > '2020-06-30']
        
        trends['covid_impact'] = {
            'pre_covid_avg': pre_covid['AQI'].mean() if len(pre_covid) > 0 else 0,
            'covid_lockdown_avg': covid_period['AQI'].mean() if len(covid_period) > 0 else 0,
            'post_covid_avg': post_covid['AQI'].mean() if len(post_covid) > 0 else 0,
            'lockdown_reduction': None
        }
        
        if len(pre_covid) > 0 and len(covid_period) > 0:
            reduction = ((pre_covid['AQI'].mean() - covid_period['AQI'].mean()) / pre_covid['AQI'].mean()) * 100
            trends['covid_impact']['lockdown_reduction'] = round(reduction, 2)
        
        return trends
    
    def _generate_critical_states_analysis(self, df, analyzer):
        """Generate analysis of critical states"""
        critical_analysis = {}
        
        # Get critical states
        critical_states = analyzer.identify_critical_states(df)
        critical_analysis['top_critical_states'] = critical_states.head(10)
        
        # State-wise category distribution
        state_categories = pd.crosstab(df['State'], df['AQI_Category'])
        state_categories['Total'] = state_categories.sum(axis=1)
        
        # Calculate percentages
        for category in self.aqi_categories.keys():
            if category in state_categories.columns:
                state_categories[f'{category}_Pct'] = (state_categories[category] / state_categories['Total'] * 100).round(2)
        
        critical_analysis['state_category_distribution'] = state_categories
        
        return critical_analysis
    
    def _generate_seasonal_analysis(self, df):
        """Generate seasonal pattern analysis"""
        seasonal_analysis = {}
        
        # Seasonal averages
        seasonal_avg = df.groupby(['Year', 'Season'])['AQI'].agg(['mean', 'std', 'count']).round(2)
        seasonal_analysis['seasonal_averages'] = seasonal_avg
        
        # Monthly patterns
        monthly_patterns = df.groupby('Month')['AQI'].agg(['mean', 'std', 'min', 'max']).round(2)
        seasonal_analysis['monthly_patterns'] = monthly_patterns
        
        # Critical months identification
        worst_months = monthly_patterns.sort_values('mean', ascending=False).head(3)
        best_months = monthly_patterns.sort_values('mean', ascending=True).head(3)
        
        seasonal_analysis['worst_months'] = worst_months
        seasonal_analysis['best_months'] = best_months
        
        return seasonal_analysis
    
    def _generate_health_impact(self, df):
        """Generate health impact assessment"""
        health_impact = {}
        
        # Category distribution over years
        category_yearly = pd.crosstab(df['Year'], df['AQI_Category'])
        category_yearly['Total'] = category_yearly.sum(axis=1)
        
        # Calculate percentages
        for category in self.aqi_categories.keys():
            if category in category_yearly.columns:
                category_yearly[f'{category}_Pct'] = (category_yearly[category] / category_yearly['Total'] * 100).round(2)
        
        health_impact['yearly_category_distribution'] = category_yearly
        
        # Population exposure estimation (simplified)
        # Assuming average population exposure based on AQI categories
        category_health_risk = {
            'Good': 'Minimal Risk',
            'Satisfactory': 'Low Risk', 
            'Moderate': 'Moderate Risk',
            'Poor': 'High Risk',
            'Very Poor': 'Very High Risk',
            'Severe': 'Hazardous'
        }
        
        health_impact['health_risk_assessment'] = category_health_risk
        
        # Days with health advisory
        advisory_days = len(df[df['AQI_Category'].isin(['Poor', 'Very Poor', 'Severe'])])
        total_days = len(df['Date'].unique())
        health_impact['advisory_days'] = {
            'count': advisory_days,
            'percentage': round((advisory_days / total_days) * 100, 2)
        }
        
        return health_impact
    
    def _generate_recommendations(self, df, analyzer):
        """Generate data-driven recommendations"""
        recommendations = {}
        
        # Overall trend
        yearly_avg = df.groupby('Year')['AQI'].mean()
        if len(yearly_avg) > 1:
            trend = 'improving' if yearly_avg.iloc[-1] < yearly_avg.iloc[0] else 'worsening'
        else:
            trend = 'stable'
        
        # Critical states
        critical_states = analyzer.identify_critical_states(df)
        top_critical = critical_states.head(3).index.tolist()
        
        # Seasonal hotspots
        worst_season = df.groupby('Season')['AQI'].mean().idxmax()
        
        recommendations['priority_actions'] = [
            f"Focus on {', '.join(top_critical)} - states with highest average AQI",
            f"Implement targeted measures during {worst_season} season",
            "Strengthen monitoring in cities with frequent 'Severe' AQI days",
            "Develop state-specific air quality management plans"
        ]
        
        recommendations['immediate_measures'] = [
            "Enhance public transportation systems in metropolitan areas",
            "Strict enforcement of construction dust control measures",
            "Promote clean energy adoption in industrial sectors",
            "Implement waste management reforms to reduce burning"
        ]
        
        recommendations['long_term_strategies'] = [
            "Develop comprehensive electric vehicle infrastructure",
            "Invest in green cover and urban forestry programs",
            "Promote clean technology in agriculture to reduce stubble burning",
            "Strengthen inter-state coordination for regional airshed management"
        ]
        
        recommendations['monitoring_improvements'] = [
            "Expand real-time monitoring network to tier-2 and tier-3 cities",
            "Implement source apportionment studies in critical regions",
            "Develop early warning systems for pollution episodes",
            "Enhance public awareness and engagement platforms"
        ]
        
        return recommendations
    
    def create_visual_report(self, df, analyzer):
        """Create a visual report with plots"""
        report = self.generate_comprehensive_report(df, analyzer)
        
        # Create a comprehensive dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Yearly AQI Trend', 'State-wise AQI Comparison',
                'Seasonal Patterns', 'AQI Category Distribution',
                'Critical States Analysis', 'Monthly AQI Pattern'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # Yearly trend
        yearly_avg = df.groupby('Year')['AQI'].mean()
        fig.add_trace(
            go.Scatter(x=yearly_avg.index, y=yearly_avg.values, mode='lines+markers', name='Yearly Avg AQI'),
            row=1, col=1
        )
        
        # State comparison
        state_avg = df.groupby('State')['AQI'].mean().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=state_avg.index, y=state_avg.values, name='State Avg AQI'),
            row=1, col=2
        )
        
        # Seasonal patterns
        seasonal_avg = df.groupby('Season')['AQI'].mean()
        fig.add_trace(
            go.Bar(x=seasonal_avg.index, y=seasonal_avg.values, name='Seasonal Avg AQI'),
            row=2, col=1
        )
        
        # Category distribution
        category_counts = df['AQI_Category'].value_counts()
        fig.add_trace(
            go.Pie(labels=category_counts.index, values=category_counts.values, name='AQI Categories'),
            row=2, col=2
        )
        
        # Critical states (days with severe AQI)
        severe_days = df[df['AQI_Category'] == 'Severe'].groupby('State').size().sort_values(ascending=False).head(8)
        fig.add_trace(
            go.Bar(x=severe_days.index, y=severe_days.values, name='Severe AQI Days'),
            row=3, col=1
        )
        
        # Monthly pattern
        monthly_avg = df.groupby('Month')['AQI'].mean()
        fig.add_trace(
            go.Scatter(x=monthly_avg.index, y=monthly_avg.values, mode='lines+markers', name='Monthly Pattern'),
            row=3, col=2
        )
        
        fig.update_layout(height=1000, title_text="Comprehensive AQI Analysis Report", showlegend=False)
        
        return fig, report
    
    def export_report_to_html(self, df, analyzer, filename="aqi_analysis_report.html"):
        """Export the complete report to HTML format"""
        fig, report = self.create_visual_report(df, analyzer)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Indian AQI Analysis Report 2020-2025</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 40px; }}
                .metric-card {{ background: #f5f5f5; padding: 20px; margin: 10px; border-radius: 5px; }}
                .critical {{ color: #d63031; font-weight: bold; }}
                .good {{ color: #00b894; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>🌫️ Indian Air Quality Index Analysis Report</h1>
            <h2>Period: {report['executive_summary']['analysis_period']}</h2>
            
            <div class="section">
                <h3>📊 Executive Summary</h3>
                <div class="metric-card">
                    <p>Average AQI: <span class="{'critical' if report['executive_summary']['average_aqi'] > 200 else 'good'}">{report['executive_summary']['average_aqi']}</span></p>
                    <p>Critical Pollution Days: {report['executive_summary']['critical_pollution_days']} ({report['executive_summary']['critical_days_percentage']}%)</p>
                    <p>States Covered: {report['executive_summary']['states_covered']}</p>
                    <p>Cities Monitored: {report['executive_summary']['cities_covered']}</p>
                </div>
            </div>
            
            <div class="section">
                <h3>📈 Visual Analysis</h3>
                <div id="plot"></div>
            </div>
            
            <div class="section">
                <h3>💡 Key Recommendations</h3>
                <h4>Priority Actions:</h4>
                <ul>
        """
        
        for action in report['recommendations']['priority_actions']:
            html_content += f"<li>{action}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <script>
        """
        
        # Add the Plotly figure
        html_content += f"Plotly.newPlot('plot', {fig.to_json()}, {{responsive: true}});"
        
        html_content += """
            </script>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename