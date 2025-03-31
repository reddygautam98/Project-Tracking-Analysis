import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Create directory for charts
os.makedirs('public/charts', exist_ok=True)

# Set plotting style for better visualizations
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Define a custom color palette for better visualizations
COLORS = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'tertiary': '#9b59b6',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'dark': '#34495e',
    'light': '#ecf0f1',
    'success': '#2ecc71',
    'info': '#3498db'
}

# Generate sample data
np.random.seed(42)

# 1. Deadline compliance data
deadline_met = np.random.choice(['Yes', 'No'], size=500, p=[0.46, 0.54])
deadline_df = pd.DataFrame({'Met Deadline?': deadline_met})

# 2. Delay distribution data
delays = np.concatenate([
    np.random.randint(-5, 1, size=230),
    np.random.randint(1, 6, size=120),
    np.random.randint(6, 11, size=80),
    np.random.randint(11, 16, size=40),
    np.random.randint(16, 30, size=30)
])
delay_df = pd.DataFrame({'Delay': delays})

# 3. Missed reasons data
reasons = ['Scope Change', 'Resource Shortage', 'Technical Issues', 'External Factors', 'Poor Planning']
counts = [85, 65, 55, 40, 35]
reasons_df = pd.DataFrame({'Reason': reasons, 'Count': counts})

# 4. Model performance data
models = ['LOGISTIC', 'RF', 'XGB', 'LGB', 'CATBOOST', 'STACKING']
accuracies = [85.33, 84.00, 82.67, 82.67, 84.67, 86.00]
model_df = pd.DataFrame({'Model': models, 'Accuracy': accuracies})

# 5. Feature importance data
features = ['Planned Duration', 'Project Complexity', 'Start Quarter', 'Deadline Extended', 'Start Month', 'Start Day of Week']
importance = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
feature_df = pd.DataFrame({'Feature': features, 'Importance': importance})

# Generate charts

# 1. Deadline compliance pie chart
plt.figure(figsize=(10, 8))
deadline_counts = deadline_df['Met Deadline?'].value_counts()
colors = [COLORS['success'], COLORS['danger']]
wedgeprops = {'width': 0.6, 'edgecolor': 'w', 'linewidth': 2}
deadline_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors, 
                    wedgeprops=wedgeprops, startangle=90, shadow=True)
plt.title('Projects Meeting Deadlines', fontsize=16, fontweight='bold')
plt.ylabel('')
# Add a clean white circle in the middle for a donut chart effect
centre_circle = plt.Circle((0, 0), 0.3, fc='white')
plt.gca().add_patch(centre_circle)
plt.tight_layout()
plt.savefig('public/charts/projects_meeting_deadlines.png', dpi=300)
plt.close()

# 2. Delay distribution histogram
plt.figure(figsize=(12, 8))
sns.histplot(delay_df['Delay'], kde=True, bins=20, color=COLORS['tertiary'])
plt.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=2, label='Deadline')
plt.title('Distribution of Project Delays', fontsize=16, fontweight='bold')
plt.xlabel('Delay (days)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('public/charts/distribution_of_project_delays.png', dpi=300)
plt.close()

# 3. Missed reasons horizontal bar chart
plt.figure(figsize=(12, 8))
colors = sns.color_palette("viridis", len(reasons_df))
reasons_df.sort_values('Count', ascending=True).plot(kind='barh', x='Reason', y='Count', color=colors)
plt.title('Top Reasons for Missed Deadlines', fontsize=16, fontweight='bold')
plt.xlabel('Count', fontsize=14)
plt.ylabel('Reason', fontsize=14)
# Add count labels to the end of each bar
for i, v in enumerate(reasons_df.sort_values('Count', ascending=True)['Count']):
    plt.text(v + 0.5, i, str(v), va='center', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7, axis='x')
plt.tight_layout()
plt.savefig('public/charts/top_reasons_for_missed_deadlines.png', dpi=300)
plt.close()

# 4. Model performance bar chart
plt.figure(figsize=(12, 8))
colors = sns.color_palette("coolwarm", len(model_df))
model_df.plot(kind='bar', x='Model', y='Accuracy', color=colors)
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.ylim(75, 90)  # Set y-axis limits for better visualization
# Add accuracy labels on top of each bar
for i, v in enumerate(model_df['Accuracy']):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.tight_layout()
plt.savefig('public/charts/model_performance.png', dpi=300)
plt.close()

# 5. Feature importance horizontal bar chart
plt.figure(figsize=(12, 8))
colors = sns.color_palette("viridis", len(feature_df))
feature_df.sort_values('Importance', ascending=True).plot(kind='barh', x='Feature', y='Importance', color=colors)
plt.title('Feature Importance', fontsize=16, fontweight='bold')
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
# Add importance labels to the end of each bar
for i, v in enumerate(feature_df.sort_values('Importance', ascending=True)['Importance']):
    plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7, axis='x')
plt.tight_layout()
plt.savefig('public/charts/feature_importance.png', dpi=300)
plt.close()

# 6. Anomaly detection scatter plot
plt.figure(figsize=(12, 8))
# Generate sample data for anomaly detection
np.random.seed(42)
planned_duration = np.random.randint(10, 100, size=100)
normal_projects = np.random.randint(-10, 20, size=90)
anomalous_projects = np.random.randint(30, 70, size=10)

# Combine data
actual_duration = np.zeros(100)
actual_duration[:90] = planned_duration[:90] + normal_projects
actual_duration[90:] = planned_duration[90:] + anomalous_projects

is_anomaly = np.zeros(100)
is_anomaly[90:] = 1

# Plot normal projects
plt.scatter(
    planned_duration[is_anomaly == 0],
    actual_duration[is_anomaly == 0],
    c=COLORS['primary'], label='Normal', alpha=0.7, s=80
)
# Plot anomalous projects
plt.scatter(
    planned_duration[is_anomaly == 1],
    actual_duration[is_anomaly == 1],
    c=COLORS['danger'], label='Anomaly', alpha=0.9, s=100, edgecolors='black'
)
# Add reference line
plt.plot([0, max(planned_duration)], 
         [0, max(planned_duration)], 
         'g--', linewidth=2, label='Perfect Estimation')
plt.xlabel('Planned Duration (days)', fontsize=14)
plt.ylabel('Actual Duration (days)', fontsize=14)
plt.title('Anomaly Detection in Project Durations', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('public/charts/anomaly_detection.png', dpi=300)
plt.close()

# 7. Forecast chart
plt.figure(figsize=(14, 8))
# Generate sample data for forecasting
dates = pd.date_range(start='2024-01-01', periods=24, freq='M')
actual_values = np.zeros(24)
forecast_values = np.zeros(24)
upper_bound = np.zeros(24)
lower_bound = np.zeros(24)

# Past data (actual)
for i in range(12):
    base_value = 50 + np.sin(i * 0.5) * 10
    random_variation = np.random.uniform(-7.5, 7.5)
    actual_values[i] = base_value + random_variation

# Future data (forecast)
last_actual = actual_values[11]
for i in range(12, 24):
    trend = (i - 11) * 1.5
    seasonal = np.sin((i) * 0.5) * 10
    forecast_values[i] = last_actual + trend + seasonal
    upper_bound[i] = forecast_values[i] + (10 + (i - 11) * 0.5)
    lower_bound[i] = max(0, forecast_values[i] - (10 + (i - 11) * 0.5))

# Create DataFrame
forecast_df = pd.DataFrame({
    'Date': dates,
    'Actual': actual_values,
    'Forecast': forecast_values,
    'Upper': upper_bound,
    'Lower': lower_bound
})

# Set actual values to NaN for future dates
forecast_df.loc[12:, 'Actual'] = np.nan
# Set forecast values to NaN for past dates
forecast_df.loc[:11, 'Forecast'] = np.nan
forecast_df.loc[:11, 'Upper'] = np.nan
forecast_df.loc[:11, 'Lower'] = np.nan

# Plot
plt.plot(forecast_df['Date'][:12], forecast_df['Actual'][:12], 'o-', color=COLORS['primary'], linewidth=2, label='Actual')
plt.plot(forecast_df['Date'][11:], forecast_df['Forecast'][11:], 'o--', color=COLORS['secondary'], linewidth=2, label='Forecast')
plt.fill_between(forecast_df['Date'][11:], forecast_df['Lower'][11:], forecast_df['Upper'][11:], color=COLORS['tertiary'], alpha=0.2, label='Confidence Interval')
plt.title('Project Completions Forecast', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Completions', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('public/charts/forecast_overview.png', dpi=300)
plt.close()

# 8. Forecast components chart
plt.figure(figsize=(14, 8))
# Generate sample data for forecast components
trend = np.zeros(24)
seasonal = np.zeros(24)
weekly = np.zeros(24)

for i in range(24):
    trend[i] = 40 + i * 1.5
    seasonal[i] = np.sin(i * 0.5) * 10
    weekly[i] = np.cos(i * 2) * 5

# Create DataFrame
components_df = pd.DataFrame({
    'Date': dates,
    'Trend': trend,
    'Seasonal': seasonal,
    'Weekly': weekly
})

# Plot
plt.plot(components_df['Date'], components_df['Trend'], '-', color=COLORS['primary'], linewidth=2, label='Trend')
plt.plot(components_df['Date'], components_df['Seasonal'], '-', color=COLORS['warning'], linewidth=2, label='Seasonal')
plt.plot(components_df['Date'], components_df['Weekly'], '-', color=COLORS['tertiary'], linewidth=2, label='Weekly Pattern')
plt.title('Forecast Components', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Component Value', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('public/charts/forecast_components.png', dpi=300)
plt.close()

print("All charts have been generated successfully!")

