import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import DataLoader
from utils.visualizer import AQIVisualizer
from utils.predictor import AQIPredictor
from utils.analyzer import AQIAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import config

# Page configuration
st.set_page_config(
    page_title="Indian AQI Analysis 2020-2025",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .critical-aqi {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .good-aqi {
        background-color: #51cf66;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .prediction-card {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class AQIApp:
    def __init__(self):
        self.data_loader = DataLoader()
        self.visualizer = AQIVisualizer()
        self.predictor = AQIPredictor()
        self.analyzer = AQIAnalyzer()
        self.df = None
    
    def run(self):
        # Header
        st.markdown('<h1 class="main-header">🌫️ Indian AQI Analysis 2020-2025</h1>', unsafe_allow_html=True)
        st.markdown("### Comprehensive Air Quality Analysis and Prediction Platform")
        
        # Sidebar
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.selectbox(
            "Choose Analysis Mode",
            ["📊 Data Overview", "📈 Trend Analysis", "🔍 State-wise Analysis", 
             "🌍 Geographical View", "🔮 AQI Prediction", "📋 Summary Report"]
        )
        
        # File upload
        st.sidebar.markdown("---")
        st.sidebar.subheader("Data Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your AQI CSV file (2020-2025)",
            type=['csv'],
            help="CSV should contain columns: Date, State, City, AQI"
        )
        
        # Sample data option
        if st.sidebar.button("📁 Generate Sample Data"):
            self.df = self.data_loader.create_sample_2020_2025_data()
            st.sidebar.success("Sample 2020-2025 data generated successfully!")
        
        # Load data
        if uploaded_file is not None:
            self.df, error = self.data_loader.load_data(uploaded_file)
            if error:
                st.error(f"Error loading data: {error}")
            else:
                st.sidebar.success(f"✅ Data loaded successfully! {len(self.df)} records (2020-2025)")
        
        # Main app logic
        if self.df is not None:
            if app_mode == "📊 Data Overview":
                self.show_data_overview()
            elif app_mode == "📈 Trend Analysis":
                self.show_trend_analysis()
            elif app_mode == "🔍 State-wise Analysis":
                self.show_state_analysis()
            elif app_mode == "🌍 Geographical View":
                self.show_geographical_analysis()
            elif app_mode == "🔮 AQI Prediction":
                self.show_prediction()
            else:
                self.show_summary_report()
        else:
            self.show_welcome_screen()
    
    def show_welcome_screen(self):
        """Welcome screen when no data is loaded"""
        st.markdown("""
        ## 🇮🇳 Welcome to Indian AQI Analysis Platform 2020-2025
        
        This comprehensive platform helps you analyze air quality data across Indian states 
        for the period **2020-2025**, including the unique COVID-19 lockdown period.
        
        ### 🎯 Key Features:
        - **📊 Data Overview**: Explore your dataset and basic statistics
        - **📈 Trend Analysis**: Yearly and seasonal trends from 2020-2025
        - **🔍 State-wise Analysis**: Compare states and identify critical regions
        - **🌍 Geographical View**: Interactive maps of AQI distribution
        - **🔮 AQI Prediction**: Forecast AQI levels for the next week
        - **📋 Summary Report**: Comprehensive analysis report
        
        ### 🚀 Getting Started:
        1. **Upload your CSV data** using the sidebar
        2. **Or generate sample data** to explore the platform
        3. **Navigate** through different analysis modes
        
        ### 📁 Expected CSV Format:
        Your CSV should contain at least these columns:
        - `Date` (YYYY-MM-DD format)
        - `State` (Indian state name)
        - `City` (City name) 
        - `AQI` (Air Quality Index value)
        
        Optional columns for enhanced analysis:
        - `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3` (Pollutant levels)
        - `Temperature`, `Humidity`, `Wind_Speed` (Weather data)
        """)
        
        # Show sample data structure
        st.subheader("📋 Sample Data Structure")
        sample_df = self.data_loader.create_sample_2020_2025_data().head(10)
        st.dataframe(sample_df)
        
        # Download sample template
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv,
            file_name="indian_aqi_2020_2025_template.csv",
            mime="text/csv",
            help="Download a sample CSV template to understand the required format"
        )
    
    def show_data_overview(self):
        """Data overview and basic statistics"""
        st.markdown('<h2 class="section-header">📊 Data Overview 2020-2025</h2>', unsafe_allow_html=True)
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(self.df):,}")
        with col2:
            st.metric("States Covered", self.df['State'].nunique())
        with col3:
            st.metric("Cities Covered", self.df['City'].nunique())
        with col4:
            date_range = f"{self.df['Date'].min().date()} to {self.df['Date'].max().date()}"
            st.metric("Date Range", date_range)
        
        # Yearly summary
        st.subheader("📅 Yearly Data Summary")
        yearly_summary = self.df.groupby('Year').agg({
            'AQI': ['mean', 'min', 'max', 'count']
        }).round(2)
        yearly_summary.columns = ['Avg AQI', 'Min AQI', 'Max AQI', 'Records']
        st.dataframe(yearly_summary)
        
        # Data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(self.df.head(100))
        
        # AQI distribution
        st.subheader("📊 AQI Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_counts = self.df['AQI_Category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title='AQI Category Distribution (2020-2025)',
                color=category_counts.index,
                color_discrete_map=config.AQI_CATEGORIES
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Seasonal pattern
            fig = self.visualizer.plot_seasonal_analysis(self.df)
            st.plotly_chart(fig, use_container_width=True)
    
    def show_trend_analysis(self):
        """Trend analysis across 2020-2025"""
        st.markdown('<h2 class="section-header">📈 Trend Analysis 2020-2025</h2>', unsafe_allow_html=True)
        
        # Yearly trends
        st.subheader("📈 Yearly AQI Trends")
        fig = self.visualizer.plot_yearly_trends(self.df)
        st.plotly_chart(fig, use_container_width=True)
        
        # State comparison heatmap
        st.subheader("🔥 State-wise AQI Comparison")
        fig = self.visualizer.plot_state_comparison(self.df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Critical periods analysis
        st.subheader("🚨 Critical Periods Analysis")
        fig = self.visualizer.plot_critical_periods(self.df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category evolution
        st.subheader("🔄 AQI Category Evolution")
        fig = self.visualizer.plot_category_evolution(self.df)
        st.plotly_chart(fig, use_container_width=True)
    
    def show_state_analysis(self):
        """State-wise detailed analysis"""
        st.markdown('<h2 class="section-header">🔍 State-wise Analysis</h2>', unsafe_allow_html=True)
        
        # State selection
        selected_states = st.multiselect(
            "Select States for Analysis",
            options=sorted(self.df['State'].unique()),
            default=['Delhi', 'Uttar Pradesh', 'Maharashtra', 'Karnataka']
        )
        
        if not selected_states:
            st.warning("Please select at least one state")
            return
        
        filtered_df = self.df[self.df['State'].isin(selected_states)]
        
        # State comparison
        st.subheader("📊 State Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            state_avg = filtered_df.groupby('State')['AQI'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=state_avg.values,
                y=state_avg.index,
                orientation='h',
                title='Average AQI by State',
                labels={'x': 'Average AQI', 'y': 'State'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Worst performing states
            critical_states = self.analyzer.identify_critical_states(filtered_df)
            st.subheader("🚨 Critical States Summary")
            st.dataframe(critical_states.head(10))
        
        # Yearly trends for selected states
        st.subheader("📈 Yearly Trends for Selected States")
        yearly_state = filtered_df.groupby(['Year', 'State'])['AQI'].mean().reset_index()
        fig = px.line(
            yearly_state,
            x='Year',
            y='AQI',
            color='State',
            title='Yearly AQI Trends for Selected States',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_geographical_analysis(self):
        """Geographical analysis with maps"""
        st.markdown('<h2 class="section-header">🌍 Geographical Analysis</h2>', unsafe_allow_html=True)
        
        # Year selection for geographical view
        year = st.selectbox(
            "Select Year for Geographical View",
            options=sorted(self.df['Year'].unique(), reverse=True)
        )
        
        # Geographical distribution
        st.subheader(f"🗺️ Geographical AQI Distribution ({year})")
        fig = self.visualizer.plot_geographical_distribution(self.df, year)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not generate geographical plot for the selected year")
        
        # Pollutant breakdown
        st.subheader(f"🌫️ Pollutant Breakdown ({year})")
        fig = self.visualizer.plot_pollutant_breakdown(self.df, year)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pollutant data not available for detailed breakdown")
    
    def show_prediction(self):
        """AQI prediction for next week"""
        st.markdown('<h2 class="section-header">🔮 AQI Prediction - Next 7 Days</h2>', unsafe_allow_html=True)
        
        # Location selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_state = st.selectbox(
                "Select State",
                options=sorted(self.df['State'].unique())
            )
        
        with col2:
            state_cities = self.df[self.df['State'] == selected_state]['City'].unique()
            selected_city = st.selectbox(
                "Select City",
                options=sorted(state_cities)
            )
        
        if st.button("🚀 Train Model & Predict", type="primary"):
            with st.spinner("Training prediction model..."):
                # Prepare features
                X, y, dates, error = self.predictor.prepare_features(self.df, selected_state, selected_city)
                
                if error:
                    st.error(f"Error preparing data: {error}")
                    return
                
                # Train model
                metrics = self.predictor.train_model(X, y)
                
                if metrics:
                    st.success(f"✅ Model trained successfully with {self.predictor.model_name}!")
                    
                    # Show model performance
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", f"{metrics['r2']:.3f}")
                    with col2:
                        st.metric("MAE", f"{metrics['mae']:.2f}")
                    with col3:
                        st.metric("RMSE", f"{metrics['rmse']:.2f}")
                    
                    # Make predictions
                    predictions, pred_error = self.predictor.predict_next_week(
                        self.df, selected_state, selected_city, days=7
                    )
                    
                    if pred_error:
                        st.error(f"Prediction error: {pred_error}")
                    else:
                        st.subheader("📅 Next 7 Days AQI Predictions")
                        
                        # Display predictions in a nice format
                        for _, pred in predictions.iterrows():
                            category_color = config.AQI_CATEGORIES.get(pred['AQI_Category'], {}).get('color', '#000000')
                            
                            st.markdown(f"""
                            <div class="prediction-card">
                                <strong>📅 {pred['Date'].strftime('%Y-%m-%d')}</strong> | 
                                <strong>AQI: {pred['Predicted_AQI']}</strong> | 
                                <span style="color: {category_color}; font-weight: bold;">{pred['AQI_Category']}</span> |
                                Confidence: {pred['Confidence']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Feature importance
                        st.subheader("🔍 Feature Importance")
                        feature_importance = self.predictor.get_feature_importance()
                        if feature_importance is not None:
                            fig = px.bar(
                                feature_importance.head(10),
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title='Top 10 Most Important Features for Prediction'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error("❌ Model training failed. Please try with different location or more data.")
    
    def show_summary_report(self):
        """Comprehensive summary report"""
        st.markdown('<h2 class="section-header">📋 Comprehensive Summary Report 2020-2025</h2>', unsafe_allow_html=True)
        
        # Overall trends
        st.subheader("📈 Overall Trends 2020-2025")
        yearly_trend = self.analyzer.analyze_trends(self.df)
        st.dataframe(yearly_trend)
        
        # Period comparison
        st.subheader("🕰️ Historical Period Comparison")
        period_comparison = self.analyzer.compare_periods(self.df)
        st.dataframe(period_comparison)
        
        # Critical states analysis
        st.subheader("🚨 Critical States Identification")
        critical_states = self.analyzer.identify_critical_states(self.df)
        st.dataframe(critical_states)
        
        # Health impact analysis
        st.subheader("🏥 Health Impact Analysis")
        health_impact = self.analyzer.generate_health_impact_report(self.df)
        st.dataframe(health_impact)
        
        # Key findings
        st.subheader("🔑 Key Findings")
        
        # Calculate some key metrics
        avg_aqi_2020 = self.df[self.df['Year'] == 2020]['AQI'].mean()
        avg_aqi_2025 = self.df[self.df['Year'] == 2025]['AQI'].mean()
        worst_state = self.df.groupby('State')['AQI'].mean().idxmax()
        best_state = self.df.groupby('State')['AQI'].mean().idxmin()
        severe_days = len(self.df[self.df['AQI_Category'] == 'Severe'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average AQI 2020", f"{avg_aqi_2020:.1f}")
            st.metric("Worst Performing State", worst_state)
            st.metric("Total Severe AQI Days", severe_days)
        
        with col2:
            st.metric("Average AQI 2025", f"{avg_aqi_2025:.1f}")
            st.metric("Best Performing State", best_state)
            change_pct = ((avg_aqi_2025 - avg_aqi_2020) / avg_aqi_2020 * 100)
            st.metric("Change 2020-2025", f"{change_pct:+.1f}%")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if avg_aqi_2025 > avg_aqi_2020:
            st.error("""
            **🚨 Urgent Action Required:**
            - AQI levels are increasing from 2020 to 2025
            - Implement stricter emission controls
            - Enhance public transportation systems
            - Promote clean energy initiatives
            """)
        else:
            st.success("""
            **✅ Positive Trend Detected:**
            - AQI levels are improving from 2020 to 2025
            - Continue current pollution control measures
            - Focus on maintaining and accelerating improvements
            - Share best practices with other states
            """)

# Run the app
if __name__ == "__main__":
    app = AQIApp()
    app.run()