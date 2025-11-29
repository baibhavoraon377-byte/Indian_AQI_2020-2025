# Configuration settings for Indian AQI Analysis 2020-2025

# Indian states and major cities
INDIAN_STATES = {
    'Delhi': ['Anand Vihar', 'RK Puram', 'Punjabi Bagh', 'Mandir Marg', 'Bawana'],
    'Uttar Pradesh': ['Noida', 'Ghaziabad', 'Lucknow', 'Kanpur', 'Varanasi', 'Agra'],
    'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik'],
    'West Bengal': ['Kolkata', 'Howrah', 'Durgapur', 'Asansol'],
    'Karnataka': ['Bengaluru', 'Mysuru', 'Hubli', 'Mangalore'],
    'Tamil Nadu': ['Chennai', 'Coimbatore', 'Madurai', 'Salem'],
    'Rajasthan': ['Jaipur', 'Udaipur', 'Jodhpur', 'Kota', 'Ajmer'],
    'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot'],
    'Punjab': ['Chandigarh', 'Ludhiana', 'Amritsar', 'Jalandhar'],
    'Haryana': ['Gurugram', 'Faridabad', 'Panipat', 'Ambala'],
    'Bihar': ['Patna', 'Gaya', 'Bhagalpur', 'Muzaffarpur'],
    'Madhya Pradesh': ['Bhopal', 'Indore', 'Jabalpur', 'Gwalior'],
    'Andhra Pradesh': ['Visakhapatnam', 'Vijayawada', 'Guntur', 'Tirupati'],
    'Telangana': ['Hyderabad', 'Warangal', 'Nizamabad', 'Khammam'],
    'Kerala': ['Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Thrissur'],
    'Odisha': ['Bhubaneswar', 'Cuttack', 'Rourkela', 'Sambalpur'],
    'Jharkhand': ['Ranchi', 'Jamshedpur', 'Dhanbad', 'Bokaro'],
    'Assam': ['Guwahati', 'Silchar', 'Dibrugarh', 'Jorhat']
}

# AQI Categories with colors and health impacts
AQI_CATEGORIES = {
    'Good': {'min': 0, 'max': 50, 'color': '#00E400', 'health_impact': 'Minimal impact'},
    'Satisfactory': {'min': 51, 'max': 100, 'color': '#87CEEB', 'health_impact': 'Minor breathing discomfort'},
    'Moderate': {'min': 101, 'max': 200, 'color': '#FFFF00', 'health_impact': 'Breathing discomfort to sensitive people'},
    'Poor': {'min': 201, 'max': 300, 'color': '#FF7E00', 'health_impact': 'Breathing discomfort to most people'},
    'Very Poor': {'min': 301, 'max': 400, 'color': '#FF0000', 'health_impact': 'Respiratory illness on prolonged exposure'},
    'Severe': {'min': 401, 'max': 500, 'color': '#8B0000', 'health_impact': 'Affects healthy people and seriously impacts those with existing diseases'}
}

# Critical months for analysis
CRITICAL_MONTHS = {
    'Stubble Burning Season': [10, 11],  # Oct-Nov
    'Winter Pollution': [12, 1, 2],      # Dec-Feb
    'Summer': [3, 4, 5],                 # Mar-May
    'Monsoon': [6, 7, 8, 9]              # Jun-Sep
}

# Analysis periods
ANALYSIS_PERIODS = {
    'COVID Lockdown': ('2020-03-25', '2020-06-30'),
    'Post-COVID Recovery': ('2020-07-01', '2021-03-31'),
    'Economic Revival': ('2021-04-01', '2022-12-31'),
    'Recent Period': ('2023-01-01', '2025-12-31')
}