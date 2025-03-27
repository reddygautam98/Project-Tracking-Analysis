# Comprehensive Project Tracking Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.linear_model import LogisticRegression
import networkx as nx
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier
import shap
import warnings
from catboost import CatBoostClassifier
from pycaret.classification import setup, compare_models
from lightgbm import LGBMClassifier
from tsfresh import extract_features
from pmdarima import auto_arima
import optuna
from mlxtend.classifier import StackingClassifier
import h2o
from h2o.automl import H2OAutoML
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Fixed fetch_data function to handle both URLs and local file paths
def fetch_data(source):
    # Check if source is a URL or local file path
    if source.startswith('http'):
        # Handle URL
        response = requests.get(source)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")
    else:
        # Handle local file path - directly use pandas to read the file
        try:
            return pd.read_csv(source)
        except Exception as e:
            raise Exception(f"Failed to read local file: {e}")

# Local file path - using the specified path
local_path = r"C:\Users\reddy\Downloads\Project Tracking Analysis\Project_Tracker_Fixed.csv"

# Load data using the fixed function
print(f"Loading data from local file: {local_path}")
df = fetch_data(local_path)
print("Data loaded successfully!")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Data Cleaning and Preprocessing
def clean_data(df):
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['Start Date', 'Deadline', 'Completion Date', 'New Deadline']
    for col in date_columns:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Convert 'Days Taken' to numeric
    df_clean['Days Taken'] = pd.to_numeric(df_clean['Days Taken'], errors='coerce')
    
    # Fill missing values in 'Met Deadline?' based on comparison of Completion Date and Deadline
    mask = df_clean['Met Deadline?'].isna()
    df_clean.loc[mask, 'Met Deadline?'] = np.where(
        df_clean.loc[mask, 'Completion Date'] <= df_clean.loc[mask, 'Deadline'], 
        'Yes', 'No'
    )
    
    # Create additional features
    df_clean['Delay Days'] = (df_clean['Completion Date'] - df_clean['Deadline']).dt.days
    df_clean['Delay Days'] = df_clean['Delay Days'].apply(lambda x: max(0, x))
    
    # Calculate project duration in days
    df_clean['Planned Duration'] = (df_clean['Deadline'] - df_clean['Start Date']).dt.days
    df_clean['Actual Duration'] = (df_clean['Completion Date'] - df_clean['Start Date']).dt.days
    
    # Calculate deadline extension in days
    df_clean['Deadline Extension'] = (df_clean['New Deadline'] - df_clean['Deadline']).dt.days
    
    # Create a binary target variable
    df_clean['Met Deadline Binary'] = df_clean['Met Deadline?'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Extract month and year for time series analysis
    df_clean['Start Month'] = df_clean['Start Date'].dt.month
    df_clean['Start Year'] = df_clean['Start Date'].dt.year
    df_clean['Completion Month'] = df_clean['Completion Date'].dt.month
    df_clean['Completion Year'] = df_clean['Completion Date'].dt.year
    
    # Create a date index for time series
    df_clean['Completion YearMonth'] = df_clean['Completion Date'].dt.to_period('M')
    
    return df_clean

# Clean the data
df_clean = clean_data(df)

print("\nData cleaning completed!")
print("\nSummary of cleaned data:")
print(df_clean.info())

# Check for missing values
print("\nMissing values in each column:")
print(df_clean.isna().sum())

# Fill remaining missing values
df_clean['Reason Missed'] = df_clean['Reason Missed'].fillna('Not Applicable')
df_clean['New Deadline'] = df_clean['New Deadline'].fillna(df_clean['Deadline'])
df_clean['Deadline Extension'] = df_clean['Deadline Extension'].fillna(0)

# 1. DESCRIPTIVE ANALYSIS
print("\n" + "="*50)
print("1. DESCRIPTIVE ANALYSIS")
print("="*50)

# Summary statistics
print("\nSummary Statistics:")
print(df_clean[['Days Taken', 'Planned Duration', 'Actual Duration', 'Delay Days', 'Deadline Extension']].describe())

# Distribution of project completion status
deadline_status = df_clean['Met Deadline?'].value_counts()
print("\nDistribution of deadline status:")
print(deadline_status)
deadline_pct = deadline_status / deadline_status.sum() * 100
print(f"\nPercentage of projects meeting deadline: {deadline_pct['Yes']:.2f}%")
print(f"Percentage of projects missing deadline: {deadline_pct['No']:.2f}%")

# Visualize project completion status
plt.figure(figsize=(10, 6))
sns.countplot(x='Met Deadline?', data=df_clean, palette='viridis')
plt.title('Distribution of Project Deadline Status')
plt.xlabel('Met Deadline?')
plt.ylabel('Count')
plt.show()

# Distribution of reasons for missing deadlines
if 'No' in deadline_status.index:
    missed_reasons = df_clean[df_clean['Met Deadline?'] == 'No']['Reason Missed'].value_counts()
    print("\nReasons for missing deadlines:")
    print(missed_reasons)
    
    plt.figure(figsize=(12, 6))
    sns.countplot(y='Reason Missed', data=df_clean[df_clean['Met Deadline?'] == 'No'], 
                 order=missed_reasons.index, palette='viridis')
    plt.title('Reasons for Missing Deadlines')
    plt.xlabel('Count')
    plt.ylabel('Reason')
    plt.tight_layout()
    plt.show()

# Distribution of days taken to complete projects
plt.figure(figsize=(12, 6))
sns.histplot(df_clean['Days Taken'].dropna(), bins=20, kde=True, color='purple')
plt.title('Distribution of Days Taken to Complete Projects')
plt.xlabel('Days Taken')
plt.ylabel('Frequency')
plt.axvline(df_clean['Days Taken'].mean(), color='red', linestyle='--', label=f'Mean: {df_clean["Days Taken"].mean():.2f} days')
plt.axvline(df_clean['Days Taken'].median(), color='green', linestyle='--', label=f'Median: {df_clean["Days Taken"].median():.2f} days')
plt.legend()
plt.show()

# Compare planned vs actual duration
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Planned Duration', y='Actual Duration', hue='Met Deadline?', data=df_clean, palette='viridis')
plt.plot([0, df_clean[['Planned Duration', 'Actual Duration']].max().max()], 
         [0, df_clean[['Planned Duration', 'Actual Duration']].max().max()], 
         'r--', label='Perfect Estimation Line')
plt.title('Planned vs Actual Project Duration')
plt.xlabel('Planned Duration (days)')
plt.ylabel('Actual Duration (days)')
plt.legend()
plt.show()

# 2. DIAGNOSTIC ANALYSIS
print("\n" + "="*50)
print("2. DIAGNOSTIC ANALYSIS")
print("="*50)

# Analyze factors affecting deadline adherence
print("\nAverage delay by reason:")
avg_delay_by_reason = df_clean.groupby('Reason Missed')['Delay Days'].mean().sort_values(ascending=False)
print(avg_delay_by_reason)

# Visualize average delay by reason
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_delay_by_reason.index, y=avg_delay_by_reason.values, palette='viridis')
plt.title('Average Delay Days by Reason')
plt.xlabel('Reason for Missing Deadline')
plt.ylabel('Average Delay (days)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Analyze relationship between planned duration and deadline adherence
plt.figure(figsize=(12, 6))
sns.boxplot(x='Met Deadline?', y='Planned Duration', data=df_clean, palette='viridis')
plt.title('Planned Duration by Deadline Status')
plt.xlabel('Met Deadline?')
plt.ylabel('Planned Duration (days)')
plt.show()

# Correlation between planned duration and actual duration
correlation = df_clean['Planned Duration'].corr(df_clean['Actual Duration'])
print(f"\nCorrelation between planned and actual duration: {correlation:.4f}")

# Analyze deadline extension patterns
plt.figure(figsize=(12, 6))
sns.boxplot(x='Reason Missed', y='Deadline Extension', data=df_clean[df_clean['Met Deadline?'] == 'No'], palette='viridis')
plt.title('Deadline Extension by Reason')
plt.xlabel('Reason for Missing Deadline')
plt.ylabel('Deadline Extension (days)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. PREDICTIVE ANALYSIS
print("\n" + "="*50)
print("3. PREDICTIVE ANALYSIS")
print("="*50)

# Prepare data for prediction
def prepare_data_for_prediction(df):
    # Select features and target
    features = df[['Planned Duration', 'Start Month', 'Start Year']]
    
    # Add reason missed as a feature (one-hot encoded)
    reason_dummies = pd.get_dummies(df['Reason Missed'], prefix='Reason')
    features = pd.concat([features, reason_dummies], axis=1)
    
    # Target variable: whether deadline will be met
    target = df['Met Deadline Binary']
    
    return features, target

# Split data for training and testing
features, target = prepare_data_for_prediction(df_clean.dropna(subset=['Met Deadline Binary']))
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='viridis')
plt.title('Top 10 Features for Predicting Deadline Adherence')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
# Removed as it has been moved to the top of the file

# Explain the model's predictions
explainer = shap.Explainer(rf_model, X_train)
shap_values = explainer(X_test)

# Visualize feature importance
shap.summary_plot(shap_values, X_test)

# Train an XGBoost classifier
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\nXGBoost Model Accuracy: {accuracy_xgb:.4f}")

# Train a CatBoost classifier
catboost_model = CatBoostClassifier(iterations=100, learning_rate=0.1, random_state=42, verbose=0)
catboost_model.fit(X_train, y_train)

# Evaluate the model
y_pred_catboost = catboost_model.predict(X_test)
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
print(f"\nCatBoost Model Accuracy: {accuracy_catboost:.4f}")

# Train a LightGBM classifier
lgbm_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train, y_train)

# Evaluate the model
y_pred_lgbm = lgbm_model.predict(X_test)
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
print(f"\nLightGBM Model Accuracy: {accuracy_lgbm:.4f}")

# Create a stacking classifier
stack_model = StackingClassifier(classifiers=[rf_model, xgb_model], meta_classifier=LogisticRegression())
stack_model.fit(X_train, y_train)

# Evaluate the model
y_pred_stack = stack_model.predict(X_test)
accuracy_stack = accuracy_score(y_test, y_pred_stack)
print(f"\nStacking Model Accuracy: {accuracy_stack:.4f}")

# Predict project duration
def prepare_data_for_duration_prediction(df):
    # Select features
    features = df[['Planned Duration', 'Start Month', 'Start Year']]
    
    # Add reason missed as a feature (one-hot encoded)
    reason_dummies = pd.get_dummies(df['Reason Missed'], prefix='Reason')
    features = pd.concat([features, reason_dummies], axis=1)
    
    # Target variable: actual duration
    target = df['Actual Duration']
    
    return features, target

# Train a model to predict actual duration
duration_features, duration_target = prepare_data_for_duration_prediction(df_clean.dropna(subset=['Actual Duration']))
X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(
    duration_features, duration_target, test_size=0.3, random_state=42
)

# Train a Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_dur, y_train_dur)

# Make predictions
y_pred_dur = rf_regressor.predict(X_test_dur)

# Evaluate the model
mse = mean_squared_error(y_test_dur, y_pred_dur)
rmse = np.sqrt(mse)
print("\nRandom Forest Regressor for Duration Prediction:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# Visualize actual vs predicted duration
plt.figure(figsize=(10, 6))
plt.scatter(y_test_dur, y_pred_dur, alpha=0.5)
plt.plot([y_test_dur.min(), y_test_dur.max()], [y_test_dur.min(), y_test_dur.max()], 'r--')
plt.title('Actual vs Predicted Project Duration')
plt.xlabel('Actual Duration (days)')
plt.ylabel('Predicted Duration (days)')
plt.show()

# Set up the PyCaret environment
clf_setup = setup(data=df_clean, target='Met Deadline Binary', silent=True)

# Compare models and select the best one
best_model = compare_models()

# Optimize Random Forest hyperparameters using Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(X_train, y_train)
    return accuracy_score(y_test, rf_model.predict(X_test))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"Best parameters: {study.best_params}")

# Initialize H2O
h2o.init()

# Convert data to H2O frame
h2o_df = h2o.H2OFrame(df_clean)

# Run AutoML
aml = H2OAutoML(max_models=10, seed=42)
aml.train(x=['Planned Duration', 'Start Month', 'Start Year'], y='Met Deadline Binary', training_frame=h2o_df)

# View leaderboard
print(aml.leaderboard)

# 4. PRESCRIPTIVE ANALYSIS
print("\n" + "="*50)
print("4. PRESCRIPTIVE ANALYSIS")
print("="*50)

# Analyze optimal project duration planning
optimal_buffer_analysis = pd.DataFrame({
    'Planned Duration': df_clean['Planned Duration'],
    'Actual Duration': df_clean['Actual Duration'],
    'Buffer Needed': df_clean['Actual Duration'] - df_clean['Planned Duration'],
    'Met Deadline?': df_clean['Met Deadline?']
})

print("\nBuffer Analysis:")
buffer_by_status = optimal_buffer_analysis.groupby('Met Deadline?')['Buffer Needed'].describe()
print(buffer_by_status)

# Calculate recommended buffer percentage
successful_projects = optimal_buffer_analysis[optimal_buffer_analysis['Met Deadline?'] == 'Yes']
failed_projects = optimal_buffer_analysis[optimal_buffer_analysis['Met Deadline?'] == 'No']

# Calculate the 75th percentile of buffer needed for failed projects
if not failed_projects.empty:
    recommended_buffer_pct = failed_projects['Buffer Needed'].quantile(0.75) / failed_projects['Planned Duration'].mean()
    print(f"\nRecommended buffer percentage: {recommended_buffer_pct:.2%}")
    
    # Apply the buffer to see how many projects would have met the deadline
    optimal_buffer_analysis['Adjusted Deadline'] = df_clean['Start Date'] + pd.to_timedelta(
        df_clean['Planned Duration'] * (1 + recommended_buffer_pct), unit='D'
    )
    optimal_buffer_analysis['Would Meet Adjusted?'] = optimal_buffer_analysis['Adjusted Deadline'] >= df_clean['Completion Date']
    
    improvement = optimal_buffer_analysis['Would Meet Adjusted?'].mean() - (df_clean['Met Deadline?'] == 'Yes').mean()
    print(f"Potential improvement in deadline adherence: {improvement:.2%}")

# Analyze common patterns in missed deadlines
if not df_clean[df_clean['Met Deadline?'] == 'No'].empty:
    print("\nCommon patterns in missed deadlines:")
    
    # Group by reason and analyze
    reason_analysis = df_clean[df_clean['Met Deadline?'] == 'No'].groupby('Reason Missed').agg({
        'Delay Days': ['mean', 'median', 'count'],
        'Planned Duration': ['mean', 'median'],
        'Actual Duration': ['mean', 'median']
    })
    
    print(reason_analysis)
    
    # Recommendations based on reasons
    print("\nPrescriptive Recommendations:")
    for reason in reason_analysis.index:
        avg_delay = reason_analysis.loc[reason, ('Delay Days', 'mean')]
        count = reason_analysis.loc[reason, ('Delay Days', 'count')]
        
        print(f"\nFor projects at risk of '{reason}':")
        print(f"- Add a buffer of at least {avg_delay:.1f} days")
        print(f"- This reason affected {count:.0f} projects")
        
        if reason == 'Scope Change':
            print("- Implement stricter change control procedures")
            print("- Ensure all stakeholders sign off on initial requirements")
        elif reason == 'Resource Constraints':
            print("- Conduct resource capacity planning before project initiation")
            print("- Consider having backup resources identified")
        elif reason == 'Technical Issues':
            print("- Include technical spike/exploration phase before committing to deadlines")
            print("- Ensure technical debt is addressed regularly")
        elif reason == 'Dependencies':
            print("- Map all dependencies before project start")
            print("- Add buffer specifically for external dependencies")

# 5. RISK ANALYSIS
print("\n" + "="*50)
print("5. RISK ANALYSIS")
print("="*50)

# Create a risk score for projects
def calculate_risk_score(row):
    score = 0
    
    # Base score on planned duration
    if row['Planned Duration'] > df_clean['Planned Duration'].quantile(0.75):
        score += 2  # Longer projects have higher risk
    
    # Add score based on historical performance of similar projects
    similar_projects = df_clean[
        (df_clean['Planned Duration'] >= row['Planned Duration'] * 0.8) &
        (df_clean['Planned Duration'] <= row['Planned Duration'] * 1.2)
    ]
    
    if not similar_projects.empty:
        failure_rate = (similar_projects['Met Deadline?'] == 'No').mean()
        score += failure_rate * 5  # Scale up to make it significant
    
    # Add score based on start month (if there's seasonality)
    month_failure_rates = df_clean.groupby('Start Month')['Met Deadline Binary'].agg(
        lambda x: 1 - x.mean()
    )
    if row['Start Month'] in month_failure_rates.index:
        score += month_failure_rates[row['Start Month']] * 3
    
    return score

# Apply risk scoring
df_clean['Risk Score'] = df_clean.apply(calculate_risk_score, axis=1)

# Categorize risk levels
df_clean['Risk Level'] = pd.cut(
    df_clean['Risk Score'], 
    bins=[0, 1, 3, 5, float('inf')],
    labels=['Low', 'Medium', 'High', 'Very High']
)

print("\nRisk Level Distribution:")
risk_distribution = df_clean['Risk Level'].value_counts()
print(risk_distribution)

# Visualize risk distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Risk Level', data=df_clean, palette='YlOrRd')
plt.title('Distribution of Project Risk Levels')
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.show()

# Analyze risk factors
print("\nAverage Risk Score by Reason Missed:")
risk_by_reason = df_clean.groupby('Reason Missed')['Risk Score'].mean().sort_values(ascending=False)
print(risk_by_reason)

# Visualize risk by reason
plt.figure(figsize=(12, 6))
sns.barplot(x=risk_by_reason.index, y=risk_by_reason.values, palette='YlOrRd')
plt.title('Average Risk Score by Reason')
plt.xlabel('Reason for Missing Deadline')
plt.ylabel('Average Risk Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Risk matrix: Impact (Delay Days) vs Probability (based on historical data)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='Delay Days', 
    y='Risk Score',
    hue='Reason Missed',
    size='Planned Duration',
    sizes=(50, 200),
    alpha=0.7,
    data=df_clean[df_clean['Met Deadline?'] == 'No']
)
plt.title('Risk Matrix: Impact vs Probability')
plt.xlabel('Impact (Delay Days)')
plt.ylabel('Risk Score (Probability Proxy)')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 6. SENTIMENT ANALYSIS
print("\n" + "="*50)
print("6. SENTIMENT ANALYSIS")
print("="*50)

# For this dataset, we'll analyze the sentiment of the "Reason Missed" field
# Since we don't have actual text comments, we'll categorize reasons into positive/negative/neutral
def categorize_sentiment(reason):
    negative_terms = ['constraint', 'issue', 'change', 'delay', 'depend']
    neutral_terms = ['not applicable']
    
    reason_lower = reason.lower()
    
    if any(term in reason_lower for term in negative_terms):
        return 'Negative'
    elif any(term in reason_lower for term in neutral_terms):
        return 'Neutral'
    else:
        return 'Positive'

# Apply sentiment categorization
df_clean['Reason Sentiment'] = df_clean['Reason Missed'].apply(categorize_sentiment)

print("\nSentiment Distribution in Reasons:")
sentiment_distribution = df_clean['Reason Sentiment'].value_counts()
print(sentiment_distribution)

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Reason Sentiment', data=df_clean, palette='viridis')
plt.title('Sentiment Distribution in Reasons')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Analyze relationship between sentiment and delay
plt.figure(figsize=(10, 6))
sns.boxplot(x='Reason Sentiment', y='Delay Days', data=df_clean, palette='viridis')
plt.title('Delay Days by Reason Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Delay Days')
plt.show()

# 7. NETWORK ANALYSIS
print("\n" + "="*50)
print("7. NETWORK ANALYSIS")
print("="*50)

# Create a network graph of projects and reasons
G = nx.Graph()

# Add nodes for projects and reasons
for _, row in df_clean.iterrows():
    project_node = f"Project: {row['Project Name']}"
    G.add_node(project_node, type='project', met_deadline=row['Met Deadline?'])
    
    if row['Met Deadline?'] == 'No' and pd.notna(row['Reason Missed']):
        reason_node = f"Reason: {row['Reason Missed']}"
        G.add_node(reason_node, type='reason')
        G.add_edge(project_node, reason_node, weight=row['Delay Days'])

# Calculate network metrics
print("\nNetwork Analysis:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Find most connected reasons
reason_connections = {node: len(list(G.neighbors(node))) 
                     for node in G.nodes() 
                     if 'type' in G.nodes[node] and G.nodes[node]['type'] == 'reason'}

if reason_connections:
    most_connected_reasons = sorted(reason_connections.items(), key=lambda x: x[1], reverse=True)
    print("\nMost connected reasons:")
    for reason, connections in most_connected_reasons[:5]:
        print(f"{reason}: {connections} projects")

# Visualize the network
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)

# Draw project nodes
project_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'project']
met_deadline_nodes = [node for node in project_nodes if G.nodes[node]['met_deadline'] == 'Yes']
missed_deadline_nodes = [node for node in project_nodes if G.nodes[node]['met_deadline'] == 'No']

nx.draw_networkx_nodes(G, pos, nodelist=met_deadline_nodes, node_color='green', node_size=100, alpha=0.8, label='Met Deadline')
nx.draw_networkx_nodes(G, pos, nodelist=missed_deadline_nodes, node_color='red', node_size=100, alpha=0.8, label='Missed Deadline')

# Draw reason nodes
reason_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'reason']
nx.draw_networkx_nodes(G, pos, nodelist=reason_nodes, node_color='blue', node_size=200, alpha=0.8, label='Reason')

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.5)

# Add labels to reason nodes only (to avoid cluttering)
reason_labels = {node: node.split(': ')[1] for node in reason_nodes}
nx.draw_networkx_labels(G, pos, labels=reason_labels, font_size=8)

plt.title('Network of Projects and Reasons for Missing Deadlines')
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.show()

# 8. TIME SERIES ANALYSIS
print("\n" + "="*50)
print("8. TIME SERIES ANALYSIS")
print("="*50)

# Prepare time series data
# Group by completion month and count projects
monthly_completions = df_clean.groupby('Completion YearMonth').size()
monthly_completions.index = monthly_completions.index.to_timestamp()
monthly_completions = monthly_completions.sort_index()

# Calculate monthly deadline adherence rate
monthly_adherence = df_clean.groupby('Completion YearMonth')['Met Deadline Binary'].mean()
monthly_adherence.index = monthly_adherence.index.to_timestamp()
monthly_adherence = monthly_adherence.sort_index()

# Plot time series
plt.figure(figsize=(14, 6))
monthly_completions.plot(marker='o')
plt.title('Number of Completed Projects Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Projects')
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 6))
monthly_adherence.plot(marker='o')
plt.title('Monthly Deadline Adherence Rate')
plt.xlabel('Date')
plt.ylabel('Adherence Rate')
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()

# Decompose time series if we have enough data points
if len(monthly_completions) >= 12:  # Need at least a year of data for seasonal decomposition
    try:
        # Decompose the time series
        decomposition = seasonal_decompose(monthly_completions, model='additive', period=12)
        
        # Plot decomposition
        plt.figure(figsize=(14, 10))
        plt.subplot(411)
        plt.plot(decomposition.observed)
        plt.title('Observed')
        plt.subplot(412)
        plt.plot(decomposition.trend)
        plt.title('Trend')
        plt.subplot(413)
        plt.plot(decomposition.seasonal)
        plt.title('Seasonal')
        plt.subplot(414)
        plt.plot(decomposition.resid)
        plt.title('Residual')
        plt.tight_layout()
        plt.show()
        
        print("\nTime Series Decomposition completed successfully.")
    except Exception as e:
        print(f"\nCould not perform time series decomposition: {e}")
else:
    print("\nNot enough data points for seasonal decomposition (need at least 12 months).")

# Extract features from time series data
time_series_features = extract_features(df_clean, column_id='Project ID', column_sort='Completion Date')
print(time_series_features.head())

# 9. FORECASTING
print("\n" + "="*50)
print("9. FORECASTING")
print("="*50)

# Forecast future project completions using ARIMA if we have enough data
if len(monthly_completions) >= 12:
    try:
        # Fit ARIMA model
        model = ARIMA(monthly_completions, order=(1, 1, 1))
        model_fit = model.fit()
        
        # Summary of the model
        print("\nARIMA Model Summary:")
        print(model_fit.summary())
        
        # Forecast next 6 months
        forecast_steps = 6
        forecast = model_fit.forecast(steps=forecast_steps)
        
        # Create forecast index
        last_date = monthly_completions.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
        forecast_series = pd.Series(forecast, index=forecast_index)
        
        # Plot forecast
        plt.figure(figsize=(14, 6))
        plt.plot(monthly_completions, label='Historical')
        plt.plot(forecast_series, label='Forecast', color='red')
        plt.title('Project Completions Forecast (Next 6 Months)')
        plt.xlabel('Date')
        plt.ylabel('Number of Projects')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print("\nForecast for the next 6 months:")
        for date, value in zip(forecast_index, forecast):
            print(f"{date.strftime('%Y-%m')}: {value:.2f} projects")
        
        # Forecast deadline adherence rate
        if len(monthly_adherence) >= 12:
            adherence_model = ARIMA(monthly_adherence, order=(1, 0, 1))
            adherence_model_fit = adherence_model.fit()
            
            # Forecast adherence
            adherence_forecast = adherence_model_fit.forecast(steps=forecast_steps)
            adherence_forecast_series = pd.Series(adherence_forecast, index=forecast_index)
            
            # Plot adherence forecast
            plt.figure(figsize=(14, 6))
            plt.plot(monthly_adherence, label='Historical')
            plt.plot(adherence_forecast_series, label='Forecast', color='red')
            plt.title('Deadline Adherence Rate Forecast (Next 6 Months)')
            plt.xlabel('Date')
            plt.ylabel('Adherence Rate')
            plt.legend()
            plt.grid(True)
            plt.axhline(y=0.5, color='green', linestyle='--', label='50% Threshold')
            plt.show()
            
            print("\nDeadline Adherence Rate Forecast for the next 6 months:")
            for date, value in zip(forecast_index, adherence_forecast):
                print(f"{date.strftime('%Y-%m')}: {value:.2%}")
    except Exception as e:
        print(f"\nCould not perform forecasting: {e}")
else:
    print("\nNot enough data points for reliable forecasting (need at least 12 months).")

# Automatically find the best ARIMA model
arima_model = auto_arima(monthly_completions, seasonal=True, m=12)
print(arima_model.summary())

# Forecast future values
forecast = arima_model.predict(n_periods=6)
print(f"Forecast: {forecast}")

# 10. CONCLUSION AND RECOMMENDATIONS
print("\n" + "="*50)
print("10. CONCLUSION AND RECOMMENDATIONS")
print("="*50)

# Overall statistics
total_projects = len(df_clean)
on_time_projects = (df_clean['Met Deadline?'] == 'Yes').sum()
delayed_projects = (df_clean['Met Deadline?'] == 'No').sum()
on_time_rate = on_time_projects / total_projects if total_projects > 0 else 0

print(f"\nTotal Projects Analyzed: {total_projects}")
print(f"Projects Completed On Time: {on_time_projects} ({on_time_rate:.2%})")
print(f"Projects Delayed: {delayed_projects} ({1-on_time_rate:.2%})")

# Average statistics
avg_planned_duration = df_clean['Planned Duration'].mean()
avg_actual_duration = df_clean['Actual Duration'].mean()
avg_delay = df_clean['Delay Days'].mean()

print(f"\nAverage Planned Duration: {avg_planned_duration:.2f} days")
print(f"Average Actual Duration: {avg_actual_duration:.2f} days")
print(f"Average Delay: {avg_delay:.2f} days")

# Top reasons for delays
if not df_clean[df_clean['Met Deadline?'] == 'No'].empty:
    top_reasons = df_clean[df_clean['Met Deadline?'] == 'No']['Reason Missed'].value_counts().head(3)
    print("\nTop 3 Reasons for Delays:")
    for reason, count in top_reasons.items():
        print(f"- {reason}: {count} projects ({count/delayed_projects:.2%} of delayed projects)")

# Key recommendations
print("\nKey Recommendations:")
print("1. Planning and Estimation:")
print("   - Add a buffer of at least 20% to project timelines based on historical data")
print("   - Consider project complexity and past performance when estimating")
print("   - Break down large projects into smaller, manageable phases")

print("\n2. Risk Management:")
print("   - Implement early warning systems for high-risk projects")
print("   - Conduct regular risk assessments throughout the project lifecycle")
print("   - Develop contingency plans for common delay reasons")

print("\n3. Process Improvements:")
print("   - Strengthen change control procedures to minimize scope changes")
print("   - Improve resource allocation and capacity planning")
print("   - Enhance dependency management and tracking")

print("\n4. Monitoring and Control:")
print("   - Implement regular project status reviews")
print("   - Track leading indicators of potential delays")
print("   - Establish clear escalation paths for issues")

# Final visualization: Project performance dashboard
plt.figure(figsize=(15, 10))

# Subplot 1: Deadline adherence pie chart
plt.subplot(2, 2, 1)
plt.pie([on_time_projects, delayed_projects], 
        labels=['Met Deadline', 'Missed Deadline'],
        autopct='%1.1f%%',
        colors=['green', 'red'],
        startangle=90)
plt.title('Project Deadline Adherence')

# Subplot 2: Top reasons for delays
plt.subplot(2, 2, 2)
if not df_clean[df_clean['Met Deadline?'] == 'No'].empty:
    reasons_count = df_clean[df_clean['Met Deadline?'] == 'No']['Reason Missed'].value_counts()
    ax = reasons_count.plot(kind='bar', color='orange')
    plt.title('Top Reasons for Delays')
    plt.xlabel('Reason')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

# Subplot 3: Planned vs Actual Duration

plt.subplot(2, 2, 3)
x = np.arange(min(len(df_clean), 20))  # Show max 20 projects for clarity
indices = df_clean.index[:min(len(df_clean), 20)]
width = 0.35

plt.bar(x - width/2, df_clean.loc[indices, 'Planned Duration'], width, label='Planned')
plt.bar(x + width/2, df_clean.loc[indices, 'Actual Duration'], width, label='Actual')

plt.title('Planned vs Actual Duration (Sample Projects)')
plt.xlabel('Project Index')
plt.ylabel('Duration (days)')
plt.legend()
plt.xticks(x)  # Ensure x-axis labels are shown properly


# Subplot 4: Risk distribution
plt.subplot(2, 2, 4)
risk_counts = df_clean['Risk Level'].value_counts()
ax = risk_counts.plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(risk_counts))))
plt.title('Project Risk Distribution')

plt.tight_layout()
plt.show()

print("\nAnalysis completed successfully!")