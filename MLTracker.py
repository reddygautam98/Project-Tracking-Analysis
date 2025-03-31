# Enhanced MLTracker - Advanced Project Tracking & Analysis System
# Adding: Automated Reports, Interactive Dashboards, Anomaly Detection, NLP, 
# Prophet Forecasting, XAI, Real-Time Monitoring, Scenario Simulation, and Alerts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap
import os
import datetime
from typing import Tuple, Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from prophet import Prophet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time

# Create directories for outputs
os.makedirs('charts', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('dashboards', exist_ok=True)
os.makedirs('alerts', exist_ok=True)

# Suppress warnings
warnings.filterwarnings('ignore')

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

# Define a custom colormap for heatmaps
custom_cmap = sns.diverging_palette(230, 20, as_cmap=True)

print("===== Enhanced MLTracker - Advanced Project Tracking & Analysis System =====")
print("Initializing advanced project analytics system...\n")

# ----------------------
# 1. DATA COLLECTION & INGESTION
# ----------------------
print("1. DATA COLLECTION & INGESTION")

def fetch_data() -> pd.DataFrame:
    """
    Load the dataset from a fixed local file path.
    
    Returns:
    pandas.DataFrame: Loaded project tracking data
    """
    file_path = r"C:\Users\reddy\Downloads\Project Tracking Analysis\Project_Tracker_Fixed.csv"
    print(f"Loading data from: {file_path}")
    
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error: {e}")
        print(f"Error loading file from {file_path}: {e}")
        print("Exiting due to missing or invalid dataset.")
        raise

# Load the dataset
try:
    df = fetch_data()
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Display basic information about the dataset
print("\nDataset Overview:")
print(f"Number of records: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# ----------------------
# 2. DATA CLEANING & FEATURE ENGINEERING
# ----------------------
print("\n2. DATA CLEANING & FEATURE ENGINEERING")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive data preprocessing and feature engineering
    
    Parameters:
    df (pandas.DataFrame): Raw project tracking data
    
    Returns:
    pandas.DataFrame: Processed data with engineered features
    """
    print("Performing data cleaning and feature engineering...")
    
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['Start Date', 'Deadline', 'Completion Date', 'New Deadline']
    for col in date_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
    
    print("✓ Converted date columns to datetime format")
    
    # Convert 'Days Taken' to numeric
    if 'Days Taken' in df_processed.columns:
        df_processed['Days Taken'] = pd.to_numeric(df_processed['Days Taken'], errors='coerce')
    
    # Calculate additional features
    print("Creating derived features...")
    
    # Time-based features
    if 'Start Date' in df_processed.columns and 'Deadline' in df_processed.columns:
        df_processed['Planned Duration'] = (df_processed['Deadline'] - df_processed['Start Date']).dt.days
    
    if 'Start Date' in df_processed.columns and 'Completion Date' in df_processed.columns:
        df_processed['Actual Duration'] = (df_processed['Completion Date'] - df_processed['Start Date']).dt.days
    
    if 'Completion Date' in df_processed.columns and 'Deadline' in df_processed.columns:
        df_processed['Delay'] = (df_processed['Completion Date'] - df_processed['Deadline']).dt.days
    
    # Deadline extension features
    if 'New Deadline' in df_processed.columns and 'Deadline' in df_processed.columns:
        df_processed['Deadline Extended'] = df_processed['New Deadline'].notna().astype(int)
        df_processed['Extension Days'] = (df_processed['New Deadline'] - df_processed['Deadline']).dt.days
    
    # Create a binary target variable for deadline met (1) or missed (0)
    if 'Met Deadline?' in df_processed.columns:
        df_processed['Deadline Met Binary'] = df_processed['Met Deadline?'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Extract temporal features for seasonality analysis
    if 'Start Date' in df_processed.columns:
        df_processed['Start Month'] = df_processed['Start Date'].dt.month
        df_processed['Start Year'] = df_processed['Start Date'].dt.year
        df_processed['Start Quarter'] = df_processed['Start Date'].dt.quarter
        df_processed['Start Day of Week'] = df_processed['Start Date'].dt.dayofweek
    
    if 'Completion Date' in df_processed.columns:
        df_processed['Completion Month'] = df_processed['Completion Date'].dt.month
        df_processed['Completion Year'] = df_processed['Completion Date'].dt.year
        df_processed['Completion Quarter'] = df_processed['Completion Date'].dt.quarter
    
    # Create efficiency ratio (planned vs actual duration)
    if 'Planned Duration' in df_processed.columns and 'Actual Duration' in df_processed.columns:
        df_processed['Duration Ratio'] = np.where(
            df_processed['Planned Duration'] > 0,
            df_processed['Actual Duration'] / df_processed['Planned Duration'],
            np.nan
        )
    
    # Create categorical features for modeling
    if 'Delay' in df_processed.columns:
        df_processed['Delay Category'] = pd.cut(
            df_processed['Delay'], 
            bins=[-float('inf'), 0, 7, 14, float('inf')],
            labels=['On Time', 'Minor Delay', 'Moderate Delay', 'Severe Delay']
        )
    
    # Create project complexity proxy based on planned duration
    if 'Planned Duration' in df_processed.columns:
        try:
            df_processed['Project Complexity'] = pd.qcut(
                df_processed['Planned Duration'].fillna(df_processed['Planned Duration'].median()), 
                q=3, 
                labels=['Low', 'Medium', 'High']
            )
        except ValueError as e:
            print(f"Warning: {e}")
            # Fallback for when qcut fails (e.g., with identical values)
            df_processed['Project Complexity'] = 'Medium'
    
    # Create risk score based on multiple factors
    risk_factors = []
    
    if 'Project Complexity' in df_processed.columns:
        complexity_risk = df_processed['Project Complexity'].map({'Low': 1.0, 'Medium': 2.0, 'High': 3.0})
        if complexity_risk is not None:
            risk_factors.append(complexity_risk)
    
    if 'Duration Ratio' in df_processed.columns:
        duration_risk = pd.cut(
            df_processed['Duration Ratio'].fillna(1), 
            bins=[0, 0.8, 1.1, 1.5, float('inf')],
            labels=[1, 2, 3, 4]
        ).astype(float)
        if duration_risk is not None:
            risk_factors.append(duration_risk)
    
    if risk_factors:
        # Convert all risk factors to numeric before summing
        risk_factors = [factor.astype(float) for factor in risk_factors]
        risk_sum = risk_factors[0].copy()
        for factor in risk_factors[1:]:
            risk_sum = risk_sum.add(factor, fill_value=0)
        
        df_processed['Risk Score'] = risk_sum / len(risk_factors)
        df_processed['Risk Level'] = pd.cut(
            df_processed['Risk Score'],
            bins=[0, 1.5, 2.5, float('inf')],
            labels=['Low', 'Medium', 'High']
        )
    
    print("✓ Created derived features")
    
    # Handle missing values
    print("Handling missing values...")
    
    # For numeric columns, fill with median
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df_processed.columns:
            median_val = df_processed[col].median() if not df_processed[col].empty else 0
            df_processed[col] = df_processed[col].fillna(median_val)
    
    # For categorical columns, fill with mode
    cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col in df_processed.columns and col != 'Reason Missed':  # Keep Reason Missed as NaN for projects that met deadline
            if not df_processed[col].empty and df_processed[col].notna().any():
                mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                df_processed[col] = df_processed[col].fillna(mode_val)
    
    print("✓ Handled missing values")
    
    return df_processed

# Preprocess the data
df_processed = preprocess_data(df)
print("\nProcessed data (first 5 rows):")
print(df_processed.head())

# ----------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------
print("\n3. EXPLORATORY DATA ANALYSIS (EDA)")

def perform_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive exploratory data analysis with visualizations
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    
    Returns:
    Dict[str, Any]: Dictionary with EDA results and statistics
    """
    print("Performing exploratory data analysis...")
    
    eda_results = {}
    
    # Summary statistics
    print("\nSummary Statistics:")
    numeric_cols = ['Planned Duration', 'Actual Duration', 'Delay', 'Days Taken']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_cols:
        stats = df[numeric_cols].describe()
        print(stats)
        eda_results['summary_stats'] = stats.to_dict()
    
    # Create visualizations one by one to avoid figure loop issues
    
    # 1. Distribution of project durations
    if 'Actual Duration' in df.columns:
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        sns.histplot(df['Actual Duration'].dropna(), kde=True, bins=20, ax=ax1, color=COLORS['primary'])
        ax1.set_title('Distribution of Project Durations', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Duration (days)', fontsize=14)
        ax1.set_ylabel('Frequency', fontsize=14)
        # Add grid for better readability
        ax1.grid(True, linestyle='--', alpha=0.7)
        # Add mean line
        mean_duration = df['Actual Duration'].mean()
        ax1.axvline(mean_duration, color=COLORS['danger'], linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_duration:.1f} days')
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('charts/distribution_of_project_durations.png', dpi=300)
        plt.close(fig1)
        print("✓ Created project durations distribution chart")
        eda_results['duration_distribution'] = 'charts/distribution_of_project_durations.png'
    
    # 2. Deadline met vs missed
    if 'Met Deadline?' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        deadline_counts = df['Met Deadline?'].value_counts()
        if not deadline_counts.empty:
            colors = [COLORS['success'], COLORS['danger']]
            wedgeprops = {'width': 0.6, 'edgecolor': 'w', 'linewidth': 2}
            deadline_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2, colors=colors, 
                                wedgeprops=wedgeprops, startangle=90, shadow=True)
            ax2.set_title('Projects Meeting Deadlines', fontsize=16, fontweight='bold')
            ax2.set_ylabel('')
            # Add a clean white circle in the middle for a donut chart effect
            centre_circle = plt.Circle((0, 0), 0.3, fc='white')
            ax2.add_patch(centre_circle)
            fig2.tight_layout()
            fig2.savefig('charts/projects_meeting_deadlines.png', dpi=300)
            plt.close(fig2)
            print("✓ Created deadline compliance pie chart")
            eda_results['deadline_compliance'] = {
                'chart': 'charts/projects_meeting_deadlines.png',
                'met_percentage': (deadline_counts.get('Yes', 0) / deadline_counts.sum() * 100) if 'Yes' in deadline_counts else 0
            }
    
    # 3. Planned vs Actual Duration
    if 'Planned Duration' in df.columns and 'Actual Duration' in df.columns:
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        scatter = ax3.scatter(df['Planned Duration'], df['Actual Duration'], 
                             alpha=0.7, c=df['Delay'], cmap='coolwarm', s=80)
        max_duration = max(df['Planned Duration'].max(), df['Actual Duration'].max())
        ax3.plot([0, max_duration], [0, max_duration], 'g--', linewidth=2, label='Perfect Estimation')
        ax3.set_title('Planned vs Actual Duration', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Planned Duration (days)', fontsize=14)
        ax3.set_ylabel('Actual Duration (days)', fontsize=14)
        ax3.grid(True, linestyle='--', alpha=0.7)
        # Add colorbar to show delay scale
        cbar = plt.colorbar(scatter)
        cbar.set_label('Delay (days)', fontsize=12)
        ax3.legend()
        fig3.tight_layout()
        fig3.savefig('charts/planned_vs_actual_duration.png', dpi=300)
        plt.close(fig3)
        print("✓ Created planned vs actual duration chart")
        eda_results['duration_comparison'] = 'charts/planned_vs_actual_duration.png'
    
    # 4. Reasons for missed deadlines
    if 'Reason Missed' in df.columns:
        reason_counts = df['Reason Missed'].value_counts().head(10)
        if not reason_counts.empty:
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            # Create a colormap for the bars
            colors = sns.color_palette("viridis", len(reason_counts))
            reason_counts.plot(kind='barh', ax=ax4, color=colors)
            ax4.set_title('Top Reasons for Missed Deadlines', fontsize=16, fontweight='bold')
            ax4.set_xlabel('Count', fontsize=14)
            ax4.set_ylabel('Reason', fontsize=14)
            # Add count labels to the end of each bar
            for i, v in enumerate(reason_counts):
                ax4.text(v + 0.1, i, str(v), va='center', fontweight='bold')
            ax4.grid(True, linestyle='--', alpha=0.7, axis='x')
            fig4.tight_layout()
            fig4.savefig('charts/top_reasons_for_missed_deadlines.png', dpi=300)
            plt.close(fig4)
            print("✓ Created reasons for missed deadlines chart")
            eda_results['missed_reasons'] = {
                'chart': 'charts/top_reasons_for_missed_deadlines.png',
                'top_reason': reason_counts.index[0] if not reason_counts.empty else 'Unknown',
                'counts': reason_counts.to_dict()
            }
    
    # 5. Delay distribution
    if 'Delay' in df.columns:
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        sns.histplot(df['Delay'].dropna(), kde=True, bins=20, ax=ax5, color=COLORS['tertiary'])
        ax5.axvline(x=0, color=COLORS['danger'], linestyle='--', linewidth=2, label='Deadline')
        ax5.set_title('Distribution of Project Delays', fontsize=16, fontweight='bold')
        ax5.set_xlabel('Delay (days)', fontsize=14)
        ax5.set_ylabel('Frequency', fontsize=14)
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend()
        fig5.tight_layout()
        fig5.savefig('charts/distribution_of_project_delays.png', dpi=300)
        plt.close(fig5)
        print("✓ Created delay distribution visualization")
        eda_results['delay_distribution'] = {
            'chart': 'charts/distribution_of_project_delays.png',
            'mean_delay': df['Delay'].mean(),
            'median_delay': df['Delay'].median()
        }
    
    # 6. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        
        fig6, ax6 = plt.subplots(figsize=(14, 12))
        sns.heatmap(correlation, mask=mask, annot=True, cmap=custom_cmap, fmt=".2f", 
                   linewidths=0.5, ax=ax6, annot_kws={"size": 10})
        ax6.set_title('Correlation Matrix', fontsize=16, fontweight='bold')
        fig6.tight_layout()
        fig6.savefig('charts/correlation_matrix.png', dpi=300)
        plt.close(fig6)
        print("✓ Created correlation matrix visualization")
        eda_results['correlation_matrix'] = 'charts/correlation_matrix.png'
    
    # 7. Project completion by month
    if 'Completion Year' in df.columns and 'Completion Month' in df.columns:
        try:
            monthly_completion = df.groupby(['Completion Year', 'Completion Month']).size()
            monthly_completion.index = monthly_completion.index.map(lambda x: f"{x[0]}-{x[1]:02d}")
            
            if not monthly_completion.empty:
                fig7, ax7 = plt.subplots(figsize=(14, 7))
                # Create a colormap for the bars
                colors = sns.color_palette("coolwarm", len(monthly_completion))
                monthly_completion.plot(kind='bar', ax=ax7, color=colors)
                ax7.set_title('Project Completions by Month', fontsize=16, fontweight='bold')
                ax7.set_xlabel('Year-Month', fontsize=14)
                ax7.set_ylabel('Number of Completions', fontsize=14)
                ax7.set_xticklabels(ax7.get_xticklabels(), rotation=90)
                ax7.grid(True, linestyle='--', alpha=0.7, axis='y')
                # Add count labels to the top of each bar
                for i, v in enumerate(monthly_completion):
                    ax7.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
                fig7.tight_layout()
                fig7.savefig('charts/project_completions_by_month.png', dpi=300)
                plt.close(fig7)
                print("✓ Created monthly completions visualization")
                eda_results['monthly_completions'] = {
                    'chart': 'charts/project_completions_by_month.png',
                    'data': monthly_completion.to_dict()
                }
        except Exception as e:
            print(f"Warning in monthly completions visualization: {e}")
    
    # 8. Deadline adherence by project complexity
    if 'Project Complexity' in df.columns and 'Deadline Met Binary' in df.columns:
        try:
            deadline_by_complexity = df.groupby('Project Complexity')['Deadline Met Binary'].mean() * 100
            
            if not deadline_by_complexity.empty:
                fig8, ax8 = plt.subplots(figsize=(10, 6))
                colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]
                deadline_by_complexity.plot(kind='bar', ax=ax8, color=colors)
                ax8.set_title('Deadline Adherence by Project Complexity', fontsize=16, fontweight='bold')
                ax8.set_xlabel('Project Complexity', fontsize=14)
                ax8.set_ylabel('Percentage of Deadlines Met', fontsize=14)
                ax8.set_ylim(0, 100)
                ax8.grid(True, linestyle='--', alpha=0.7, axis='y')
                # Add percentage labels to the top of each bar
                for i, v in enumerate(deadline_by_complexity):
                    ax8.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
                fig8.tight_layout()
                fig8.savefig('charts/deadline_adherence_by_complexity.png', dpi=300)
                plt.close(fig8)
                print("✓ Created deadline adherence by complexity visualization")
                eda_results['complexity_adherence'] = {
                    'chart': 'charts/deadline_adherence_by_complexity.png',
                    'data': deadline_by_complexity.to_dict()
                }
        except Exception as e:
            print(f"Warning in deadline adherence visualization: {e}")
    
    # 9. Duration ratio boxplot
    if 'Project Complexity' in df.columns and 'Duration Ratio' in df.columns:
        try:
            fig9, ax9 = plt.subplots(figsize=(12, 6))
            palette = {"Low": COLORS['success'], "Medium": COLORS['warning'], "High": COLORS['danger']}
            sns.boxplot(x='Project Complexity', y='Duration Ratio', data=df, ax=ax9, palette=palette)
            ax9.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Perfect Estimation')
            ax9.set_title('Duration Ratio by Project Complexity', fontsize=16, fontweight='bold')
            ax9.set_xlabel('Project Complexity', fontsize=14)
            ax9.set_ylabel('Actual/Planned Duration Ratio', fontsize=14)
            ax9.grid(True, linestyle='--', alpha=0.7, axis='y')
            ax9.legend()
            fig9.tight_layout()
            fig9.savefig('charts/duration_ratio_boxplot.png', dpi=300)
            plt.close(fig9)
            print("✓ Created duration ratio boxplot")
            eda_results['duration_ratio'] = 'charts/duration_ratio_boxplot.png'
        except Exception as e:
            print(f"Warning in duration ratio boxplot: {e}")
    
    # 10. Risk level distribution
    if 'Risk Level' in df.columns:
        try:
            risk_counts = df['Risk Level'].value_counts()
            
            if not risk_counts.empty:
                fig10, ax10 = plt.subplots(figsize=(10, 6))
                colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]
                risk_counts.plot(kind='bar', ax=ax10, color=colors)
                ax10.set_title('Project Risk Level Distribution', fontsize=16, fontweight='bold')
                ax10.set_xlabel('Risk Level', fontsize=14)
                ax10.set_ylabel('Number of Projects', fontsize=14)
                ax10.grid(True, linestyle='--', alpha=0.7, axis='y')
                # Add count labels to the top of each bar
                for i, v in enumerate(risk_counts):
                    ax10.text(i, v + 1, str(v), ha='center', fontweight='bold')
                fig10.tight_layout()
                fig10.savefig('charts/risk_level_distribution.png', dpi=300)
                plt.close(fig10)
                print("✓ Created risk level distribution chart")
                eda_results['risk_distribution'] = {
                    'chart': 'charts/risk_level_distribution.png',
                    'data': risk_counts.to_dict()
                }
        except Exception as e:
            print(f"Warning in risk level visualization: {e}")
    
    print("Exploratory data analysis completed")
    return eda_results

# Perform EDA
try:
    eda_results = perform_eda(df_processed)
except Exception as e:
    print(f"Error in EDA: {e}")
    print("Continuing with analysis...")
    eda_results = {}

# ----------------------
# 4. PREDICTIVE MODELING
# ----------------------
print("\n4. PREDICTIVE MODELING")

def prepare_data_for_ml(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for machine learning models
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    
    Returns:
    tuple: X_train, X_test, y_train, y_test for classification
    """
    print("Preparing data for machine learning...")
    
    # Select relevant features for classification
    features = [
        'Planned Duration', 'Start Month', 'Start Year', 'Start Quarter', 
        'Start Day of Week', 'Project Complexity', 'Deadline Extended'
    ]
    
    # Make sure all features exist in the dataframe
    features = [f for f in features if f in df.columns]
    
    # Add 'Reason Missed' if available
    if 'Reason Missed' in df.columns:
        features.append('Reason Missed')
    
    # Select features and target
    X = df[features].copy()
    y = df['Deadline Met Binary'] if 'Deadline Met Binary' in df.columns else None
    
    if y is None:
        raise ValueError("Target variable 'Deadline Met Binary' not found in dataframe")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"✓ Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples) sets")
    
    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline for categorical and numerical features
    
    Parameters:
    X_train (pandas.DataFrame): Training features
    
    Returns:
    ColumnTransformer: Preprocessing pipeline
    """
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_train.select_dtypes(include=['number']).columns
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    transformers = []
    
    if len(numerical_cols) > 0:
        transformers.append(('num', numerical_transformer, numerical_cols))
    
    if len(categorical_cols) > 0:
        transformers.append(('cat', categorical_transformer, categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return preprocessor

def train_classification_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame,
    y_train: pd.Series, 
    y_test: pd.Series
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate multiple classification models
    
    Parameters:
    X_train, X_test, y_train, y_test: Train and test data
    
    Returns:
    dict: Trained models and their performance metrics
    """
    print("\nTraining classification models to predict deadline adherence...")
    
    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(X_train)
    
    # Define base models
    base_models = [
        ('logistic', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('xgb', xgb.XGBClassifier(random_state=42)),
        ('lgb', lgb.LGBMClassifier(random_state=42, min_data_in_leaf=5, verbose=-1, min_gain_to_split=0.01)),
        ('catboost', cb.CatBoostClassifier(random_state=42, verbose=0))
    ]
    
    # Train and evaluate each model
    model_results = {}
    
    for name, model in base_models:
        try:
            print(f"Training {name} model...")
            
            # Create pipeline with preprocessing
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Evaluate model
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            print(f"✓ {name.upper()} model trained - Accuracy: {accuracy:.4f}")
            
            # Store results
            model_results[name] = {
                'pipeline': pipeline,
                'accuracy': accuracy,
                'report': report
            }
        except Exception as e:
            print(f"Error training {name} model: {e}")
    
    # Create and train stacking classifier if we have at least 2 base models
    if len(model_results) >= 2:
        try:
            print("\nTraining stacking classifier...")
            
            # Prepare base estimators with preprocessing
            base_estimators = []
            for name, model_info in model_results.items():
                base_estimators.append((
                    name,
                    model_info['pipeline']
                    )
                )
            
            # Create stacking classifier
            stacking_clf = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(),
                cv=5
            )
            
            # Train stacking classifier
            stacking_clf.fit(X_train, y_train)
            
            # Evaluate stacking classifier
            y_pred_stack = stacking_clf.predict(X_test)
            stack_accuracy = accuracy_score(y_test, y_pred_stack)
            stack_report = classification_report(y_test, y_pred_stack)
            
            print(f"✓ Stacking classifier trained - Accuracy: {stack_accuracy:.4f}")
            
            # Store stacking results
            model_results['stacking'] = {
                'pipeline': stacking_clf,
                'accuracy': stack_accuracy,
                'report': stack_report
            }
        except Exception as e:
            print(f"Error training stacking classifier: {e}")
    
    # Find best model
    if model_results:
        best_model = max(model_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest performing model: {best_model[0].upper()} with accuracy {best_model[1]['accuracy']:.4f}")
    
    return model_results

# Prepare data for machine learning
X_train, X_test, y_train, y_test = prepare_data_for_ml(df_processed)

# Train classification models
model_results = train_classification_models(X_train, X_test, y_train, y_test)

# ----------------------
# 5. ANOMALY DETECTION
# ----------------------
print("\n5. ANOMALY DETECTION")

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies in project data using Isolation Forest
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    
    Returns:
    pd.DataFrame: DataFrame with anomaly scores and flags
    """
    print("Detecting anomalies in project data...")
    
    # Create a copy of the dataframe
    df_anomaly = df.copy()
    
    # Select features for anomaly detection
    features = ['Planned Duration', 'Actual Duration', 'Delay']
    features = [f for f in features if f in df_anomaly.columns]
    
    if len(features) < 2:
        print("Not enough features for anomaly detection")
        return df_anomaly
    
    try:
        # Prepare data
        X = df_anomaly[features].fillna(df_anomaly[features].median())
        
        # Initialize and fit the model
        isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # Assume 10% of projects are anomalies
            random_state=42
        )
        
        # Fit and predict
        df_anomaly['anomaly_score'] = isolation_forest.fit_predict(X)
        
        # Convert to binary anomaly flag (-1 for anomalies, 1 for normal)
        df_anomaly['is_anomaly'] = df_anomaly['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
        
        # Calculate anomaly severity (higher is more anomalous)
        df_anomaly['anomaly_severity'] = isolation_forest.score_samples(X)
        df_anomaly['anomaly_severity'] = -df_anomaly['anomaly_severity']  # Invert so higher = more anomalous
        
        # Normalize severity to 0-100 scale
        min_severity = df_anomaly['anomaly_severity'].min()
        max_severity = df_anomaly['anomaly_severity'].max()
        if max_severity > min_severity:
            df_anomaly['anomaly_severity'] = ((df_anomaly['anomaly_severity'] - min_severity) / 
                                             (max_severity - min_severity) * 100)
        
        # Count anomalies
        anomaly_count = df_anomaly['is_anomaly'].sum()
        print(f"✓ Detected {anomaly_count} anomalous projects ({anomaly_count/len(df_anomaly)*100:.1f}%)")
        
        # Visualize anomalies
        if 'Planned Duration' in features and 'Actual Duration' in features:
            plt.figure(figsize=(12, 8))
            # Plot normal projects
            plt.scatter(
                df_anomaly[df_anomaly['is_anomaly'] == 0]['Planned Duration'],
                df_anomaly[df_anomaly['is_anomaly'] == 0]['Actual Duration'],
                c=COLORS['primary'], label='Normal', alpha=0.7, s=80
            )
            # Plot anomalous projects
            plt.scatter(
                df_anomaly[df_anomaly['is_anomaly'] == 1]['Planned Duration'],
                df_anomaly[df_anomaly['is_anomaly'] == 1]['Actual Duration'],
                c=COLORS['danger'], label='Anomaly', alpha=0.9, s=100, edgecolors='black'
            )
            # Add reference line
            plt.plot([0, df_anomaly['Planned Duration'].max()], 
                     [0, df_anomaly['Planned Duration'].max()], 
                     'g--', linewidth=2, label='Perfect Estimation')
            plt.xlabel('Planned Duration (days)', fontsize=14)
            plt.ylabel('Actual Duration (days)', fontsize=14)
            plt.title('Anomaly Detection in Project Durations', fontsize=16, fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('charts/anomaly_detection.png', dpi=300)
            plt.close()
            print("✓ Created anomaly detection visualization")
        
        return df_anomaly
    
    except Exception as e:
        print(f"Error in anomaly detection: {e}")
        return df

# Detect anomalies
df_with_anomalies = detect_anomalies(df_processed)

# ----------------------
# 6. NLP FOR DELAY REASONS
# ----------------------
print("\n6. NLP FOR DELAY REASONS")

def analyze_delay_reasons(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze delay reasons using NLP techniques
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    
    Returns:
    Dict[str, Any]: NLP analysis results
    """
    print("Analyzing delay reasons with NLP...")
    
    # Check if we have the necessary data
    if 'Reason Missed' not in df.columns:
        print("No 'Reason Missed' column found for NLP analysis")
        return {}
    
    # Filter to only include projects that missed deadlines
    missed_df = df[df['Met Deadline?'] == 'No'].copy()
    
    if missed_df.empty or missed_df['Reason Missed'].isna().all():
        print("No delay reasons found for NLP analysis")
        return {}
    
    try:
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        # Preprocess text
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        def preprocess_text(text):
            if pd.isna(text):
                return ""
            # Tokenize
            tokens = word_tokenize(text.lower())
            # Remove stopwords and lemmatize
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
            return " ".join(tokens)
        
        # Apply preprocessing
        missed_df['processed_reason'] = missed_df['Reason Missed'].apply(preprocess_text)
        
        # Extract key terms using TF-IDF
        vectorizer = TfidfVectorizer(max_features=50)
        tfidf_matrix = vectorizer.fit_transform(missed_df['processed_reason'])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate average TF-IDF score for each term
        tfidf_means = tfidf_matrix.mean(axis=0)
        tfidf_means = np.asarray(tfidf_means).flatten()
        
        # Create a dictionary of terms and their scores
        term_scores = {term: score for term, score in zip(feature_names, tfidf_means)}
        
        # Sort terms by score
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        top_terms = sorted_terms[:10]
        
        print("\nTop terms in delay reasons:")
        for term, score in top_terms:
            print(f"  - {term}: {score:.4f}")
        
        # Topic modeling with LDA
        n_topics = min(3, len(missed_df))
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tfidf_matrix)
        
        # Get top words for each topic
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'id': topic_idx,
                'words': top_words,
                'weight': float(topic.sum() / lda.components_.sum())
            })
        
        print("\nIdentified delay reason topics:")
        for topic in topics:
            print(f"  - Topic {topic['id']+1} ({topic['weight']*100:.1f}%): {', '.join(topic['words'][:5])}")
        
        # Visualize top terms
        plt.figure(figsize=(12, 8))
        terms = [term for term, _ in top_terms]
        scores = [score for _, score in top_terms]
        # Create a colormap for the bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(terms)))
        plt.barh(terms, scores, color=colors)
        plt.xlabel('TF-IDF Score', fontsize=14)
        plt.ylabel('Term', fontsize=14)
        plt.title('Top Terms in Project Delay Reasons', fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        # Add score labels to the end of each bar
        for i, score in enumerate(scores):
            plt.text(score + 0.001, i, f"{score:.4f}", va='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig('charts/delay_reason_terms.png', dpi=300)
        plt.close()
        print("✓ Created delay reason terms visualization")
        
        # Return results
        return {
            'top_terms': top_terms,
            'topics': topics,
            'visualization': 'charts/delay_reason_terms.png'
        }
    
    except Exception as e:
        print(f"Error in NLP analysis: {e}")
        return {}

# Analyze delay reasons
nlp_results = analyze_delay_reasons(df_processed)

# ----------------------
# 7. ADVANCED FORECASTING WITH PROPHET
# ----------------------
print("\n7. ADVANCED FORECASTING WITH PROPHET")

def forecast_with_prophet(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform advanced forecasting using Prophet.

    Parameters:
    df (pd.DataFrame): Processed project tracking data

    Returns:
    Dict[str, Any]: Forecasting results including metrics and visualizations
    """
    try:
        # Ensure the required columns are present
        if 'Completion Date' not in df.columns:
            print("Error: 'Completion Date' column is missing.")
            return {}

        # Prepare the data for Prophet
        df_prophet = df[['Completion Date']].copy()
        df_prophet.rename(columns={'Completion Date': 'ds'}, inplace=True)
        df_prophet['y'] = 1  # Count each project as 1 for forecasting

        # Aggregate data by date
        df_prophet = df_prophet.groupby('ds').sum().reset_index()

        # Initialize the Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.add_seasonality(name='yearly', period=365.25, fourier_order=10)
        model.add_seasonality(name='weekly', period=7, fourier_order=3)

        # Fit the model
        model.fit(df_prophet)

        # Create a future DataFrame
        future = model.make_future_dataframe(periods=365)  # Forecast for 1 year into the future

        # Generate the forecast
        forecast = model.predict(future)

        # Save visualizations
        fig1 = model.plot(forecast)
        fig1.savefig('charts/forecast_overview.png', dpi=300)

        fig2 = model.plot_components(forecast)
        fig2.savefig('charts/forecast_components.png', dpi=300)

        print("✓ Forecasting completed successfully.")

        # Return results
        return {
            'forecast': forecast,
            'visualizations': ['charts/forecast_overview.png', 'charts/forecast_components.png'],
            'metrics': {
                'mean_forecast': forecast['yhat'].mean(),
                'total_forecast': forecast['yhat'].sum(),
                'trend_direction': 'increasing' if forecast['trend'].iloc[-1] > forecast['trend'].iloc[0] else 'decreasing'
            }
        }

    except Exception as e:
        print(f"Error in forecasting: {e}")
        return {}

# Forecast with Prophet
# Ensure the function `forecast_with_prophet` is defined before calling it
forecast_results = forecast_with_prophet(df_processed)

# Enhanced Yearly and Weekly Seasonality Charts
def plot_seasonality_components(forecast: pd.DataFrame) -> None:
    """
    Plot enhanced yearly and weekly seasonality components from the forecast.

    Parameters:
    forecast (pd.DataFrame): Forecast data from Prophet
    """
    try:
        # Ensure the required columns are present
        if 'yearly' not in forecast.columns or 'weekly' not in forecast.columns:
            print("Error: 'yearly' or 'weekly' seasonality components are missing.")
            return

        # Extract yearly and weekly seasonality
        yearly_data = forecast[['ds', 'yearly']].dropna()
        weekly_data = forecast[['ds', 'weekly']].dropna()

        # Create a figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'hspace': 0.4})

        # Trend Component
        if 'trend' in forecast.columns:
            axes[0].plot(forecast['ds'], forecast['trend'], color='blue', linewidth=2)
            axes[0].set_title('Trend Component', fontsize=16, fontweight='bold')
            axes[0].set_xlabel('Date', fontsize=14)
            axes[0].set_ylabel('Trend', fontsize=14)
            axes[0].grid(True, linestyle='--', alpha=0.7)

        # Yearly Seasonality
        axes[1].plot(yearly_data['ds'].dt.dayofyear, yearly_data['yearly'], color='green', linewidth=2)
        axes[1].fill_between(yearly_data['ds'].dt.dayofyear, yearly_data['yearly'], color='green', alpha=0.2)
        axes[1].set_title('Yearly Seasonality', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Day of Year', fontsize=14)
        axes[1].set_ylabel('Effect', fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # Weekly Seasonality
        days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_data['day_of_week'] = weekly_data['ds'].dt.dayofweek
        weekly_avg = weekly_data.groupby('day_of_week')['weekly'].mean()
        axes[2].bar(days_of_week, weekly_avg, color='orange', alpha=0.8)
        axes[2].set_title('Weekly Seasonality', fontsize=16, fontweight='bold')
        axes[2].set_xlabel('Day of Week', fontsize=14)
        axes[2].set_ylabel('Effect', fontsize=14)
        axes[2].grid(True, linestyle='--', alpha=0.7)

        # Save the chart
        plt.tight_layout()
        plt.savefig('charts/enhanced_seasonality_components.png', dpi=300)
        plt.close()
        print("✓ Created enhanced yearly and weekly seasonality charts")

    except Exception as e:
        print(f"Error in plotting seasonality components: {e}")

# Call the function to plot seasonality components
if forecast_results and 'forecast' in forecast_results:
    forecast = pd.DataFrame(forecast_results['forecast'])
    plot_seasonality_components(forecast)
else:
    print("No forecast data available for plotting seasonality components.")

# ----------------------
# 8. EXPLAINABLE AI (XAI) ENHANCEMENTS
# ----------------------
print("\n8. EXPLAINABLE AI (XAI) ENHANCEMENTS")

def explain_model_predictions(
    df: pd.DataFrame,
    model_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Explain model predictions using SHAP
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    model_results (Dict): Trained models and their performance metrics
    
    Returns:
    Dict[str, Any]: XAI results
    """
    print("Generating model explanations with SHAP...")
    
    if not model_results:
        print("No trained models available for explanation")
        return {}
    
    # Find the best model
    best_model_name = max(model_results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model_info = model_results[best_model_name]
    best_pipeline = best_model_info['pipeline']
    
    try:
        # Prepare data for explanation
        features = [
            'Planned Duration', 'Start Month', 'Start Year', 'Start Quarter', 
            'Start Day of Week', 'Project Complexity', 'Deadline Extended'
        ]
        features = [f for f in features if f in df.columns]
        
        X = df[features].copy()
        
        # Get the preprocessor and model from the pipeline
        preprocessor = best_pipeline.named_steps['preprocessor']
        model = best_pipeline.named_steps['classifier']
        
        # Transform the data
        X_transformed = preprocessor.transform(X)
        
        # Create a SHAP explainer
        if best_model_name in ['rf', 'xgb', 'lgb', 'catboost']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)
        
        # Calculate SHAP values
        shap_values = explainer(X_transformed)
        
        # Create summary plot with enhanced styling
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_transformed, plot_type="bar", show=False, color=COLORS['primary'])
        plt.title(f'Feature Importance ({best_model_name.upper()} Model)', fontsize=16, fontweight='bold')
        plt.xlabel('mean(|SHAP value|)', fontsize=14)
        plt.tight_layout()
        plt.savefig('charts/shap_feature_importance.png', dpi=300)
        plt.close()
        print("✓ Created SHAP feature importance visualization")
        
        # Create detailed SHAP plot for a sample project
        if len(shap_values) > 0:
            plt.figure(figsize=(16, 10))
            sample_idx = 0  # Use the first project as an example
            shap.plots.waterfall(shap_values[sample_idx], show=False, max_display=10)
            plt.title('Detailed Explanation for a Sample Project', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('charts/shap_waterfall_sample.png', dpi=300)
            plt.close()
            print("✓ Created SHAP waterfall plot for sample project")
        
        # Create force plot for all projects
        plt.figure(figsize=(20, 12))
        shap_values_summary = shap.Explanation(
            values=shap_values.values[:,:,1] if shap_values.values.ndim == 3 else shap_values.values,
            base_values=shap_values.base_values if shap_values.base_values.ndim == 1 else shap_values.base_values[:,1],
            data=shap_values.data,
            feature_names=shap_values.feature_names
        )
        shap.plots.beeswarm(shap_values_summary, show=False)
        plt.title('SHAP Values for All Projects', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('charts/shap_beeswarm.png', dpi=300)
        plt.close()
        print("✓ Created SHAP beeswarm plot")
        
        # Extract feature importance
        if hasattr(shap_values, 'values'):
            if shap_values.values.ndim == 3:
                # For multi-class models
                importance_values = np.abs(shap_values.values[:,:,1]).mean(0)
            else:
                # For binary classification
                importance_values = np.abs(shap_values.values).mean(0)
            
            # Get feature names
            if hasattr(shap_values, 'feature_names'):
                feature_names = shap_values.feature_names
            else:
                feature_names = [f"feature_{i}" for i in range(len(importance_values))]
            
            # Create feature importance dictionary
            feature_importance = {
                name: float(importance) for name, importance in zip(feature_names, importance_values)
            }
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            print("\nFeature importance ranking:")
            for i, (feature, importance) in enumerate(feature_importance.items(), 1):
                print(f"  {i}. {feature}: {importance:.4f}")
        else:
            feature_importance = {}
        
        # Return results
        return {
            'model': best_model_name,
            'feature_importance': feature_importance,
            'visualizations': [
                'charts/shap_feature_importance.png',
                'charts/shap_waterfall_sample.png',
                'charts/shap_beeswarm.png'
            ]
        }
    
    except Exception as e:
        print(f"Error in XAI analysis: {e}")
        return {}

# Get XAI explanations
xai_results = explain_model_predictions(df_processed, model_results)

# ----------------------
# 9. AUTOMATED REPORT GENERATION
# ----------------------
print("\n9. AUTOMATED REPORT GENERATION")

def generate_automated_report(
    df: pd.DataFrame,
    eda_results: Dict[str, Any],
    model_results: Dict[str, Dict[str, Any]],
    nlp_results: Dict[str, Any],
    forecast_results: Dict[str, Any],
    xai_results: Dict[str, Any]
) -> str:
    """
    Generate an automated HTML report with key findings
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    eda_results (Dict): EDA results
    model_results (Dict): Model results
    nlp_results (Dict): NLP analysis results
    forecast_results (Dict): Forecasting results
    xai_results (Dict): XAI results
    
    Returns:
    str: Path to the generated report
    """
    print("Generating automated project analysis report...")
    
    # Create report timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Project Tracking Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3, h4 {{
                color: #2c3e50;
            }}
            .header {{
                background-color: #34495e;
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 30px;
                border-radius: 5px;
            }}
            .section {{
                margin-bottom: 40px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin-bottom: 20px;
            }}
            .metric-box {{
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 15px;
                width: 30%;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                text-align: center;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
                margin: 10px 0;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .chart {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #7f8c8d;
                font-size: 14px;
            }}
            .alert {{
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
            }}
            .alert-success {{
                background-color: #d4edda;
                color: #155724;
            }}
            .alert-warning {{
                background-color: #fff3cd;
                color: #856404;
            }}
            .alert-danger {{
                background-color: #f8d7da;
                color: #721c24;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Project Tracking Analysis Report</h1>
            <p>Generated on {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
    """
    
    # Add executive summary metrics
    deadline_met_rate = df['Met Deadline?'].eq('Yes').mean() * 100 if 'Met Deadline?' in df.columns else 0
    avg_delay = df['Delay'].mean() if 'Delay' in df.columns else 0
    total_projects = len(df)
    
    html_content += f"""
            <div class="metric-container">
                <div class="metric-box">
                    <h3>Total Projects</h3>
                    <div class="metric-value">{total_projects}</div>
                </div>
                <div class="metric-box">
                    <h3>Deadline Compliance</h3>
                    <div class="metric-value">{deadline_met_rate:.1f}%</div>
                </div>
                <div class="metric-box">
                    <h3>Average Delay</h3>
                    <div class="metric-value">{avg_delay:.1f} days</div>
                </div>
            </div>
            
            <p>This report provides a comprehensive analysis of project tracking data, including exploratory analysis, 
            predictive modeling, anomaly detection, and forecasting. Key insights and recommendations are provided to 
            improve project management practices and outcomes.</p>
    """
    
    # Add key findings alert box
    if deadline_met_rate < 50:
        alert_class = "alert-danger"
        alert_message = "Critical: Low deadline compliance rate indicates significant project management issues."
    elif deadline_met_rate < 75:
        alert_class = "alert-warning"
        alert_message = "Warning: Moderate deadline compliance rate suggests room for improvement in project management."
    else:
        alert_class = "alert-success"
        alert_message = "Good: High deadline compliance rate indicates effective project management practices."
    
    html_content += f"""
            <div class="alert {alert_class}">
                <strong>Key Finding:</strong> {alert_message}
            </div>
        </div>
    """
    
    # Add EDA section
    html_content += """
        <div class="section">
            <h2>Exploratory Data Analysis</h2>
    """
    
    # Add deadline compliance chart if available
    if 'deadline_compliance' in eda_results and 'chart' in eda_results['deadline_compliance']:
        html_content += f"""
            <div class="chart-container">
                <h3>Project Deadline Compliance</h3>
                <img class="chart" src="../{eda_results['deadline_compliance']['chart']}" alt="Deadline Compliance Chart">
            </div>
        """
    
    # Add delay distribution chart if available
    if 'delay_distribution' in eda_results and 'chart' in eda_results['delay_distribution']:
        html_content += f"""
            <div class="chart-container">
                <h3>Distribution of Project Delays</h3>
                <img class="chart" src="../{eda_results['delay_distribution']['chart']}" alt="Delay Distribution Chart">
            </div>
        """
    
    # Add missed reasons chart if available
    if 'missed_reasons' in eda_results and 'chart' in eda_results['missed_reasons']:
        html_content += f"""
            <div class="chart-container">
                <h3>Top Reasons for Missed Deadlines</h3>
                <img class="chart" src="../{eda_results['missed_reasons']['chart']}" alt="Missed Reasons Chart">
            </div>
        """
    
    html_content += """
        </div>
    """
    
    # Add Predictive Modeling section
    if model_results:
        html_content += """
        <div class="section">
            <h2>Predictive Modeling</h2>
        """
        
        # Add model performance table
        html_content += """
            <h3>Model Performance Comparison</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Notes</th>
                </tr>
        """
        
        # Add rows for each model
        for name, info in model_results.items():
            accuracy = info.get('accuracy', 0) * 100
            html_content += f"""
                <tr>
                    <td>{name.upper()}</td>
                    <td>{accuracy:.2f}%</td>
                    <td>{name.capitalize()} classifier trained on project features</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
        
        # Add feature importance if available
        if xai_results and 'visualizations' in xai_results:
            html_content += f"""
            <div class="chart-container">
                <h3>Feature Importance Analysis</h3>
                <img class="chart" src="../{xai_results['visualizations'][0]}" alt="Feature Importance Chart">
            </div>
            """
        
        html_content += """
        </div>
        """
    
    # Add Anomaly Detection section
    html_content += """
    <div class="section">
        <h2>Anomaly Detection</h2>
        <p>The analysis identified projects that deviate significantly from normal patterns, which may require special attention.</p>
    """
    
    # Add anomaly detection chart
    html_content += """
        <div class="chart-container">
            <h3>Anomalous Projects</h3>
            <img class="chart" src="../charts/anomaly_detection.png" alt="Anomaly Detection Chart">
        </div>
    """
    
    # Add anomaly statistics
    anomaly_count = df_with_anomalies['is_anomaly'].sum() if 'is_anomaly' in df_with_anomalies.columns else 0
    anomaly_percentage = (anomaly_count / len(df_with_anomalies)) * 100 if len(df_with_anomalies) > 0 else 0
    
    html_content += f"""
        <div class="metric-container">
            <div class="metric-box">
                <h3>Anomalous Projects</h3>
                <div class="metric-value">{anomaly_count}</div>
            </div>
            <div class="metric-box">
                <h3>Percentage</h3>
                <div class="metric-value">{anomaly_percentage:.1f}%</div>
            </div>
        </div>
    </div>
    """
    
    # Add NLP Analysis section
    if nlp_results:
        html_content += """
        <div class="section">
            <h2>NLP Analysis of Delay Reasons</h2>
            <p>Natural Language Processing was used to analyze the reasons provided for project delays.</p>
        """
        
        # Add delay reason terms chart if available
        if 'visualization' in nlp_results:
            html_content += f"""
            <div class="chart-container">
                <h3>Key Terms in Delay Reasons</h3>
                <img class="chart" src="../{nlp_results['visualization']}" alt="Delay Reason Terms Chart">
            </div>
            """
        
        # Add topics if available
        if 'topics' in nlp_results:
            html_content += """
            <h3>Identified Delay Reason Topics</h3>
            <table>
                <tr>
                    <th>Topic</th>
                    <th>Weight</th>
                    <th>Key Terms</th>
                </tr>
            """
            
            for topic in nlp_results['topics']:
                html_content += f"""
                <tr>
                    <td>Topic {topic['id']+1}</td>
                    <td>{topic['weight']*100:.1f}%</td>
                    <td>{', '.join(topic['words'][:5])}</td>
                </tr>
                """
            
            html_content += """
            </table>
            """
        
        html_content += """
        </div>
        """
    
    # Add Forecasting section
    if forecast_results:
        html_content += """
        <div class="section">
            <h2>Project Completion Forecasting</h2>
            <p>Advanced time series forecasting was used to predict future project completions.</p>
        """
        
        # Add forecast chart if available
        if 'visualizations' in forecast_results:
            html_content += f"""
            <div class="chart-container">
                <h3>Project Completions Forecast</h3>
                <img class="chart" src="../{forecast_results['visualizations'][0]}" alt="Project Completions Forecast Chart">
            </div>
            
            <div class="chart-container">
                <h3>Forecast Components</h3>
                <img class="chart" src="../{forecast_results['visualizations'][1]}" alt="Forecast Components Chart">
            </div>
            """
        
        # Add forecast metrics if available
        if 'metrics' in forecast_results:
            metrics = forecast_results['metrics']
            html_content += f"""
            <div class="metric-container">
                <div class="metric-box">
                    <h3>Daily Avg (Next 30 Days)</h3>
                    <div class="metric-value">{metrics.get('mean_forecast', 0):.2f}</div>
                </div>
                <div class="metric-box">
                    <h3>Total (Next 30 Days)</h3>
                    <div class="metric-value">{metrics.get('total_forecast', 0):.1f}</div>
                </div>
                <div class="metric-box">
                    <h3>Trend</h3>
                    <div class="metric-value">{metrics.get('trend_direction', 'Unknown').capitalize()}</div>
                </div>
            </div>
            """
        
        html_content += """
        </div>
        """
    
    # Add Recommendations section
    html_content += """
    <div class="section">
        <h2>Recommendations</h2>
        <ol>
    """
    
    # Generate recommendations based on analysis results
    recommendations = []
    
    # Deadline compliance recommendation
    if deadline_met_rate < 50:
        recommendations.append("Implement a comprehensive review of project management processes to address the critical issue of missed deadlines.")
    elif deadline_met_rate < 75:
        recommendations.append("Improve deadline compliance by implementing buffer times in project planning based on historical data.")
    else:
        recommendations.append("Maintain high deadline compliance by continuing effective project management practices.")
    
    # Delay reasons recommendation
    if 'missed_reasons' in eda_results and 'top_reason' in eda_results['missed_reasons']:
        top_reason = eda_results['missed_reasons']['top_reason']
        recommendations.append(f"Address the most common reason for delays ('{top_reason}') with targeted interventions.")
    
    # Add NLP-based recommendations
    if nlp_results and 'topics' in nlp_results and nlp_results['topics']:
        top_topic_words = nlp_results['topics'][0]['words'][:3]
        recommendations.append(f"Focus on mitigating issues related to {', '.join(top_topic_words)} as identified in the NLP analysis.")
    
    # Add forecasting-based recommendations
    if forecast_results and 'metrics' in forecast_results:
        trend = forecast_results['metrics'].get('trend_direction', '')
        if trend == 'increasing':
            recommendations.append("Prepare for an increasing trend in project completions by ensuring adequate resources are available.")
        elif trend == 'decreasing':
            recommendations.append("Address the forecasted decrease in project completions by investigating potential bottlenecks.")
    
    # Add anomaly-based recommendations
    if 'is_anomaly' in df_with_anomalies.columns and df_with_anomalies['is_anomaly'].sum() > 0:
        recommendations.append("Investigate anomalous projects identified in the analysis to understand root causes and prevent similar issues.")
    
    # Add model-based recommendations
    if xai_results and 'feature_importance' in xai_results and xai_results['feature_importance']:
        top_feature = list(xai_results['feature_importance'].keys())[0]
        recommendations.append(f"Focus on optimizing '{top_feature}' as it was identified as the most important factor in project success.")
    
    # Add general recommendations
    recommendations.extend([
        "Implement a real-time monitoring system to track project progress and identify potential issues early.",
        "Use the predictive models to assess risk for new projects before they begin.",
        "Conduct regular scenario simulations to prepare for potential project disruptions."
    ])
    
    # Add recommendations to HTML
    for recommendation in recommendations:
        html_content += f"""
        <li>{recommendation}</li>
        """
    
    html_content += """
        </ol>
    </div>
    """
    
    # Add footer
    html_content += f"""
        <div class="footer">
            <p>Generated by Enhanced MLTracker - Advanced Project Tracking & Analysis System</p>
            <p>{timestamp}</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    report_filename = f"reports/project_analysis_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_filename, 'w') as f:
        f.write(html_content)
    
    print(f"✓ Generated automated report: {report_filename}")
    return report_filename

# Generate automated report
report_file = generate_automated_report(
    df_processed, 
    eda_results, 
    model_results, 
    nlp_results, 
    forecast_results, 
    xai_results
)

# ----------------------
# 10. INTERACTIVE DASHBOARDS
# ----------------------
print("\n10. INTERACTIVE DASHBOARDS")

def create_interactive_dashboards(
    df: pd.DataFrame,
    df_with_anomalies: pd.DataFrame,
    forecast_results: Dict[str, Any]
) -> List[str]:
    """
    Create interactive dashboards using Plotly
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    df_with_anomalies (pandas.DataFrame): Data with anomaly detection results
    forecast_results (Dict): Forecasting results
    
    Returns:
    List[str]: Paths to the generated dashboards
    """
    print("Creating interactive dashboards...")
    
    dashboard_files = []
    
    try:
        # 1. Project Overview Dashboard
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Deadline Compliance', 
                'Project Duration Distribution',
                'Delay Distribution',
                'Project Complexity'
            ),
            specs=[
                [{"type": "pie"}, {"type": "histogram"}],
                [{"type": "histogram"}, {"type": "pie"}]
            ]
        )
        
        # Deadline compliance pie chart
        if 'Met Deadline?' in df.columns:
            deadline_counts = df['Met Deadline?'].value_counts()
            fig1.add_trace(
                go.Pie(
                    labels=deadline_counts.index,
                    values=deadline_counts.values,
                    hole=0.4,
                    marker_colors=[COLORS['success'], COLORS['danger']]
                ),
                row=1, col=1
            )
        
        # Project duration histogram
        if 'Actual Duration' in df.columns:
            fig1.add_trace(
                go.Histogram(
                    x=df['Actual Duration'],
                    nbinsx=20,
                    marker_color=COLORS['primary']
                ),
                row=1, col=2
            )
        
        # Delay distribution histogram
        if 'Delay' in df.columns:
            fig1.add_trace(
                go.Histogram(
                    x=df['Delay'],
                    nbinsx=20,
                    marker_color=COLORS['tertiary']
                ),
                row=2, col=1
            )
        
        # Project complexity pie chart
        if 'Project Complexity' in df.columns:
            complexity_counts = df['Project Complexity'].value_counts()
            fig1.add_trace(
                go.Pie(
                    labels=complexity_counts.index,
                    values=complexity_counts.values,
                    hole=0.4,
                    marker_colors=[COLORS['success'], COLORS['warning'], COLORS['danger']]
                ),
                row=2, col=2
            )
        
        # Update layout
        fig1.update_layout(
            title_text='Project Overview Dashboard',
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        # Save dashboard
        dashboard1_path = 'dashboards/project_overview_dashboard.html'
        pio.write_html(fig1, dashboard1_path)
        dashboard_files.append(dashboard1_path)
        print(f"✓ Created project overview dashboard: {dashboard1_path}")
        
        # 2. Anomaly Detection Dashboard
        if 'is_anomaly' in df_with_anomalies.columns:
            fig2 = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Anomaly Detection (Planned vs Actual Duration)',
                    'Anomaly Severity Distribution',
                    'Anomalies by Project Complexity',
                    'Anomalies by Start Month'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "histogram"}],
                    [{"type": "bar"}, {"type": "bar"}]
                ]
            )
            
            # Anomaly detection scatter plot
            if 'Planned Duration' in df_with_anomalies.columns and 'Actual Duration' in df_with_anomalies.columns:
                # Normal projects
                normal_df = df_with_anomalies[df_with_anomalies['is_anomaly'] == 0]
                fig2.add_trace(
                    go.Scatter(
                        x=normal_df['Planned Duration'],
                        y=normal_df['Actual Duration'],
                        mode='markers',
                        name='Normal',
                        marker=dict(color=COLORS['primary'], size=8, opacity=0.6)
                    ),
                    row=1, col=1
                )
                
                # Anomalous projects
                anomaly_df = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1]
                fig2.add_trace(
                    go.Scatter(
                        x=anomaly_df['Planned Duration'],
                        y=anomaly_df['Actual Duration'],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color=COLORS['danger'], size=10, opacity=0.8)
                    ),
                    row=1, col=1
                )
                
                # Add reference line
                max_duration = max(df_with_anomalies['Planned Duration'].max(), df_with_anomalies['Actual Duration'].max())
                fig2.add_trace(
                    go.Scatter(
                        x=[0, max_duration],
                        y=[0, max_duration],
                        mode='lines',
                        name='Perfect Estimation',
                        line=dict(color=COLORS['success'], dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Anomaly severity histogram
            if 'anomaly_severity' in df_with_anomalies.columns:
                fig2.add_trace(
                    go.Histogram(
                        x=df_with_anomalies['anomaly_severity'],
                        nbinsx=20,
                        marker_color=COLORS['danger']
                    ),
                    row=1, col=2
                )
            
            # Anomalies by project complexity
            if 'Project Complexity' in df_with_anomalies.columns:
                complexity_anomalies = df_with_anomalies.groupby('Project Complexity')['is_anomaly'].sum()
                fig2.add_trace(
                    go.Bar(
                        x=complexity_anomalies.index,
                        y=complexity_anomalies.values,
                        marker_color=COLORS['tertiary']
                    ),
                    row=2, col=1
                )
            
            # Anomalies by start month
            if 'Start Month' in df_with_anomalies.columns:
                month_anomalies = df_with_anomalies.groupby('Start Month')['is_anomaly'].sum()
                fig2.add_trace(
                    go.Bar(
                        x=month_anomalies.index,
                        y=month_anomalies.values,
                        marker_color=COLORS['primary']
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig2.update_layout(
                title_text='Anomaly Detection Dashboard',
                height=800,
                template='plotly_white'
            )
            
            # Save dashboard
            dashboard2_path = 'dashboards/anomaly_detection_dashboard.html'
            pio.write_html(fig2, dashboard2_path)
            dashboard_files.append(dashboard2_path)
            print(f"✓ Created anomaly detection dashboard: {dashboard2_path}")
        
        # 3. Risk Assessment Dashboard
        if 'Risk Level' in df.columns and 'Risk Score' in df.columns:
            fig3 = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Risk Level Distribution',
                    'Risk Score Distribution',
                    'Risk Level by Project Complexity',
                    'Risk Score vs Delay'
                ),
                specs=[
                    [{"type": "pie"}, {"type": "histogram"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ]
            )
            
            # Risk level pie chart
            risk_counts = df['Risk Level'].value_counts()
            fig3.add_trace(
                go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    hole=0.4,
                    marker_colors=[COLORS['success'], COLORS['warning'], COLORS['danger']]
                ),
                row=1, col=1
            )
            
            # Risk score histogram
            fig3.add_trace(
                go.Histogram(
                    x=df['Risk Score'],
                    nbinsx=20,
                    marker_color=COLORS['primary']
                ),
                row=1, col=2
            )
            
            # Risk level by project complexity
            if 'Project Complexity' in df.columns:
                risk_by_complexity = pd.crosstab(df['Project Complexity'], df['Risk Level'])
                
                for risk_level in risk_by_complexity.columns:
                    color = COLORS['success'] if risk_level == 'Low' else (COLORS['warning'] if risk_level == 'Medium' else COLORS['danger'])
                    fig3.add_trace(
                        go.Bar(
                            x=risk_by_complexity.index,
                            y=risk_by_complexity[risk_level],
                            name=risk_level,
                            marker_color=color
                        ),
                        row=2, col=1
                    )
            
            # Risk score vs delay scatter plot
            if 'Delay' in df.columns:
                fig3.add_trace(
                    go.Scatter(
                        x=df['Risk Score'],
                        y=df['Delay'],
                        mode='markers',
                        marker=dict(
                            color=df['Risk Score'],
                            colorscale='Viridis',
                            size=10,
                            opacity=0.7
                        )
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig3.update_layout(
                title_text='Risk Assessment Dashboard',
                height=800,
                template='plotly_white'
            )
            
            # Save dashboard
            dashboard3_path = 'dashboards/risk_assessment_dashboard.html'
            pio.write_html(fig3, dashboard3_path)
            dashboard_files.append(dashboard3_path)
            print(f"✓ Created risk assessment dashboard: {dashboard3_path}")
        
        return dashboard_files
    
    except Exception as e:
        print(f"Error creating interactive dashboards: {e}")
        return dashboard_files

# Create interactive dashboards
dashboard_files = create_interactive_dashboards(df_processed, df_with_anomalies, forecast_results)

# ----------------------
# 11. REAL-TIME MONITORING
# ----------------------
print("\n11. REAL-TIME MONITORING SIMULATION")

def simulate_real_time_monitoring(df: pd.DataFrame) -> None:
    """
    Simulate real-time monitoring of project progress
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    """
    print("Simulating real-time project monitoring...")
    
    # Create a monitoring dashboard template
    monitoring_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Real-Time Project Monitoring</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .header {
                background-color: #34495e;
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 30px;
                border-radius: 5px;
            }
            .dashboard {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-bottom: 30px;
            }
            .card {
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .metric {
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
                margin: 10px 0;
            }
            .project-list {
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .status-on-track {
                color: #2ecc71;
                font-weight: bold;
            }
            .status-at-risk {
                color: #f39c12;
                font-weight: bold;
            }
            .status-delayed {
                color: #e74c3c;
                font-weight: bold;
            }
            .refresh-time {
                text-align: right;
                color: #7f8c8d;
                font-size: 14px;
                margin-bottom: 20px;
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #7f8c8d;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Real-Time Project Monitoring</h1>
            <p>Live dashboard for tracking project progress</p>
        </div>
        
        <div class="refresh-time">
            Last updated: <span id="update-time">TIMESTAMP</span>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h3>Active Projects</h3>
                <div class="metric" id="active-projects">ACTIVE_PROJECTS</div>
            </div>
            <div class="card">
                <h3>Projects At Risk</h3>
                <div class="metric" id="at-risk-projects">AT_RISK_PROJECTS</div>
            </div>
            <div class="card">
                <h3>Average Completion Rate</h3>
                <div class="metric" id="avg-completion">AVG_COMPLETION%</div>
            </div>
            <div class="card">
                <h3>Deadline Compliance (30-day)</h3>
                <div class="metric" id="deadline-compliance">DEADLINE_COMPLIANCE%</div>
            </div>
        </div>
        
        <div class="project-list">
            <h2>Active Projects Status</h2>
            <table>
                <tr>
                    <th>Project Name</th>
                    <th>Start Date</th>
                    <th>Deadline</th>
                    <th>Progress</th>
                    <th>Status</th>
                    <th>Risk Level</th>
                </tr>
                PROJECT_ROWS
            </table>
        </div>
        
        <div class="footer">
            <p>Enhanced MLTracker - Real-Time Monitoring System</p>
        </div>
    </body>
    </html>
    """
    
    # Simulate active projects (subset of the data)
    if len(df) > 10:
        active_projects = df.sample(min(10, len(df))).copy()
    else:
        active_projects = df.copy()
    
    # Generate random progress for each project
    np.random.seed(int(time.time()))
    active_projects['Progress'] = np.random.randint(10, 100, size=len(active_projects))
    
    # Determine status based on progress and deadline
    def determine_status(row):
        if 'Deadline' not in row or pd.isna(row['Deadline']):
            return 'Unknown'
        
        days_to_deadline = (row['Deadline'] - pd.Timestamp.now()).days
        progress = row['Progress']
        
        if days_to_deadline < 0:
            return 'Delayed'
        elif days_to_deadline < 7 and progress < 80:
            return 'At Risk'
        elif progress < 30 and days_to_deadline < 14:
            return 'At Risk'
        else:
            return 'On Track'
    
    active_projects['Status'] = active_projects.apply(determine_status, axis=1)
    
    # Count projects by status
    status_counts = active_projects['Status'].value_counts()
    at_risk_count = status_counts.get('At Risk', 0) + status_counts.get('Delayed', 0)
    
    # Calculate metrics
    active_count = len(active_projects)
    avg_completion = active_projects['Progress'].mean()
    deadline_compliance = (active_projects['Status'] == 'On Track').mean() * 100
    
    # Generate project rows HTML
    project_rows = ""
    for _, row in active_projects.iterrows():
        project_name = row.get('Project Name', 'Unknown')
        start_date = row.get('Start Date', pd.NaT)
        deadline = row.get('Deadline', pd.NaT)
        progress = row.get('Progress', 0)
        status = row.get('Status', 'Unknown')
        risk_level = row.get('Risk Level', 'Unknown')
        
        # Format dates
        start_date_str = start_date.strftime('%Y-%m-%d') if not pd.isna(start_date) else 'N/A'
        deadline_str = deadline.strftime('%Y-%m-%d') if not pd.isna(deadline) else 'N/A'
        
        # Determine status class
        status_class = ""
        if status == 'On Track':
            status_class = "status-on-track"
        elif status == 'At Risk':
            status_class = "status-at-risk"
        elif status == 'Delayed':
            status_class = "status-delayed"
        
        # Create row HTML
        project_rows += f"""
        <tr>
            <td>{project_name}</td>
            <td>{start_date_str}</td>
            <td>{deadline_str}</td>
            <td>{progress}%</td>
            <td class="{status_class}">{status}</td>
            <td>{risk_level}</td>
        </tr>
        """
    
    # Update the template with actual values
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    monitoring_html = monitoring_html.replace("TIMESTAMP", timestamp)
    monitoring_html = monitoring_html.replace("ACTIVE_PROJECTS", str(active_count))
    monitoring_html = monitoring_html.replace("AT_RISK_PROJECTS", str(at_risk_count))
    monitoring_html = monitoring_html.replace("AVG_COMPLETION", f"{avg_completion:.1f}")
    monitoring_html = monitoring_html.replace("DEADLINE_COMPLIANCE", f"{deadline_compliance:.1f}")
    monitoring_html = monitoring_html.replace("PROJECT_ROWS", project_rows)
    
    # Write to file
    monitoring_file = "dashboards/real_time_monitoring.html"
    with open(monitoring_file, 'w') as f:
        f.write(monitoring_html)
    
    print(f"✓ Created real-time monitoring dashboard: {monitoring_file}")

# Simulate real-time monitoring
simulate_real_time_monitoring(df_processed)

# ----------------------
# 12. SCENARIO SIMULATION
# ----------------------
print("\n12. SCENARIO SIMULATION")

def perform_scenario_simulation(
    df: pd.DataFrame,
    model_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform scenario simulations to assess impact of different factors
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    model_results (Dict): Trained models and their performance metrics
    
    Returns:
    Dict[str, Any]: Simulation results
    """
    print("Performing scenario simulations...")
    
    if not model_results:
        print("No trained models available for scenario simulation")
        return {}
    
    try:
        # Find the best model
        best_model_name = max(model_results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model_info = model_results[best_model_name]
        best_pipeline = best_model_info['pipeline']
        
        # Create base scenario from median values
        base_scenario = {}
        
        # Get feature list from the model
        features = [
            'Planned Duration', 'Start Month', 'Start Year', 'Start Quarter', 
            'Start Day of Week', 'Project Complexity', 'Deadline Extended'
        ]
        features = [f for f in features if f in df.columns]
        
        # Create base scenario with median/mode values
        for feature in features:
            if df[feature].dtype in [np.float64, np.int64]:
                base_scenario[feature] = df[feature].median()
            else:
                base_scenario[feature] = df[feature].mode()[0]
        
        # Create a DataFrame with the base scenario
        base_df = pd.DataFrame([base_scenario])
        
        # Get base prediction
        base_prediction = best_pipeline.predict_proba(base_df)[0][1]  # Probability of meeting deadline
        
        print(f"\nBase scenario (probability of meeting deadline): {base_prediction:.2f}")
        
        # Define scenarios
        scenarios = {
            "Increased Duration": {
                "description": "Increase planned duration by 20%",
                "changes": {"Planned Duration": lambda x: x * 1.2}
            },
            "Decreased Duration": {
                "description": "Decrease planned duration by 20%",
                "changes": {"Planned Duration": lambda x: x * 0.8}
            },
            "High Complexity": {
                "description": "Set project complexity to High",
                "changes": {"Project Complexity": "High"}
            },
            "Low Complexity": {
                "description": "Set project complexity to Low",
                "changes": {"Project Complexity": "Low"}
            },
            "Extended Deadline": {
                "description": "Set deadline extended to True",
                "changes": {"Deadline Extended": 1}
            },
            "Summer Start": {
                "description": "Start project in summer (June)",
                "changes": {"Start Month": 6, "Start Quarter": 2}
            },
            "Winter Start": {
                "description": "Start project in winter (January)",
                "changes": {"Start Month": 1, "Start Quarter": 1}
            }
        }
        
        # Run simulations
        simulation_results = {}
        
        for name, scenario in scenarios.items():
            # Create a copy of the base scenario
            scenario_df = base_df.copy()
            
            # Apply changes
            for feature, change in scenario["changes"].items():
                if callable(change):
                    scenario_df[feature] = change(scenario_df[feature])
                else:
                    scenario_df[feature] = change
            
            # Make prediction
            prediction = best_pipeline.predict_proba(scenario_df)[0][1]
            
            # Calculate impact
            impact = prediction - base_prediction
            
            # Store results
            simulation_results[name] = {
                "description": scenario["description"],
                "prediction": prediction,
                "impact": impact
            }
            
            print(f"Scenario '{name}': {prediction:.2f} (impact: {impact:+.2f})")
        
        # Create visualization with enhanced styling
        plt.figure(figsize=(12, 8))
        
        # Sort scenarios by impact
        sorted_scenarios = sorted(simulation_results.items(), key=lambda x: x[1]["impact"])
        
        # Extract data for plotting
        scenario_names = [s[0] for s in sorted_scenarios]
        impacts = [s[1]["impact"] for s in sorted_scenarios]
        
        # Create colors based on impact
        colors = [COLORS['danger'] if impact < 0 else COLORS['success'] for impact in impacts]
        
        # Create bar chart
        plt.barh(scenario_names, impacts, color=colors)
        
        # Add base prediction line
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Impact on Probability of Meeting Deadline', fontsize=14)
        plt.ylabel('Scenario', fontsize=14)
        plt.title('Impact of Different Scenarios on Project Success', fontsize=16, fontweight='bold')
        
        # Add values to bars
        for i, impact in enumerate(impacts):
            plt.text(
                impact + (0.01 if impact >= 0 else -0.01),
                i,
                f"{impact:+.2f}",
                va='center',
                ha='left' if impact >= 0 else 'right',
                fontweight='bold'
            )
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        plt.tight_layout()
        plt.savefig('charts/scenario_simulation_impact.png', dpi=300)
        plt.close()
        print("✓ Created scenario simulation impact visualization")
        
        # Create a more detailed visualization with actual probabilities
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        predictions = [s[1]["prediction"] for s in sorted_scenarios]
        
        # Create a gradient colormap based on prediction values
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(predictions)))
        
        # Create a horizontal bar chart
        plt.barh(scenario_names, predictions, color=colors)
        
        # Add base prediction line
        plt.axvline(x=base_prediction, color='red', linestyle='--', alpha=0.7, 
                   label=f'Base Scenario ({base_prediction:.2f})')
        
        # Add labels and title
        plt.xlabel('Probability of Meeting Deadline', fontsize=14)
        plt.ylabel('Scenario', fontsize=14)
        plt.title('Probability of Meeting Deadline Under Different Scenarios', 
                 fontsize=16, fontweight='bold')
        plt.xlim(0, 1)
        
        # Add values to bars
        for i, prob in enumerate(predictions):
            plt.text(
                prob + 0.01,
                i,
                f"{prob:.2f}",
                va='center',
                ha='left',
                fontweight='bold'
            )
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('charts/scenario_simulation_probabilities.png', dpi=300)
        plt.close()
        print("✓ Created scenario simulation probabilities visualization")
        
        return {
            "base_prediction": base_prediction,
            "scenarios": simulation_results,
            "visualizations": [
                "charts/scenario_simulation_impact.png",
                "charts/scenario_simulation_probabilities.png"
            ]
        }
    
    except Exception as e:
        print(f"Error in scenario simulation: {e}")
        return {}

# Perform scenario simulation
simulation_results = perform_scenario_simulation(df_processed, model_results)

# ----------------------
# 13. CUSTOM ALERTS AND NOTIFICATIONS
# ----------------------
print("\n13. CUSTOM ALERTS AND NOTIFICATIONS")

def generate_custom_alerts(df: pd.DataFrame, df_with_anomalies: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate custom alerts based on project data analysis
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    df_with_anomalies (pandas.DataFrame): Data with anomaly detection results
    
    Returns:
    List[Dict[str, Any]]: List of alerts
    """
    print("Generating custom alerts and notifications...")
    
    alerts = []
    
    try:
        # 1. Anomaly alerts
        if 'is_anomaly' in df_with_anomalies.columns and 'Project Name' in df_with_anomalies.columns:
            anomaly_projects = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1]
            
            if not anomaly_projects.empty:
                for _, project in anomaly_projects.iterrows():
                    project_name = project['Project Name']
                    severity = project.get('anomaly_severity', 0)
                    
                    alerts.append({
                        "type": "anomaly",
                        "level": "high" if severity > 75 else "medium",
                        "project": project_name,
                        "message": f"Anomaly detected in project '{project_name}' with severity {severity:.1f}/100",
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        # 2. Deadline alerts
        if 'Deadline' in df.columns and 'Project Name' in df.columns:
            today = pd.Timestamp.now()
            upcoming_deadlines = df[(df['Deadline'] > today) & (df['Deadline'] <= today + pd.Timedelta(days=7))]
            
            if not upcoming_deadlines.empty:
                for _, project in upcoming_deadlines.iterrows():
                    project_name = project['Project Name']
                    deadline = project['Deadline']
                    days_remaining = (deadline - today).days
                    
                    alerts.append({
                        "type": "deadline",
                        "level": "high" if days_remaining <= 3 else "medium",
                        "project": project_name,
                        "message": f"Project '{project_name}' deadline approaching in {days_remaining} days",
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        # 3. Risk alerts
        if 'Risk Level' in df.columns and 'Project Name' in df.columns:
            high_risk_projects = df[df['Risk Level'] == 'High']
            
            if not high_risk_projects.empty:
                for _, project in high_risk_projects.iterrows():
                    project_name = project['Project Name']
                    risk_score = project.get('Risk Score', 0)
                    
                    alerts.append({
                        "type": "risk",
                        "level": "high",
                        "project": project_name,
                        "message": f"Project '{project_name}' has high risk level (score: {risk_score:.1f})",
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        # 4. Resource allocation alerts
        if 'Start Date' in df.columns and 'Project Name' in df.columns:
            today = pd.Timestamp.now()
            upcoming_starts = df[(df['Start Date'] > today) & (df['Start Date'] <= today + pd.Timedelta(days=14))]
            
            if len(upcoming_starts) >= 3:
                alerts.append({
                    "type": "resource",
                    "level": "medium",
                    "project": "Multiple",
                    "message": f"Resource allocation warning: {len(upcoming_starts)} projects starting in the next 14 days",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # Generate alerts HTML
        if alerts:
            alerts_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Project Alerts and Notifications</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 800px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    h1, h2 {
                        color: #2c3e50;
                    }
                    .header {
                        background-color: #34495e;
                        color: white;
                        padding: 20px;
                        text-align: center;
                        margin-bottom: 30px;
                        border-radius: 5px;
                    }
                    .alert {
                        margin-bottom: 15px;
                        padding: 15px;
                        border-radius: 5px;
                        border-left: 5px solid;
                    }
                    .alert-high {
                        background-color: #f8d7da;
                        border-left-color: #e74c3c;
                    }
                    .alert-medium {
                        background-color: #fff3cd;
                        border-left-color: #f39c12;
                    }
                    .alert-low {
                        background-color: #d4edda;
                        border-left-color: #2ecc71;
                    }
                    .alert-title {
                        font-weight: bold;
                        margin-bottom: 5px;
                    }
                    .alert-message {
                        margin-bottom: 5px;
                    }
                    .alert-meta {
                        font-size: 12px;
                        color: #7f8c8d;
                    }
                    .footer {
                        text-align: center;
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 1px solid #eee;
                        color: #7f8c8d;
                        font-size: 14px;
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Project Alerts and Notifications</h1>
                    <p>Generated on """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                </div>
                
                <h2>Active Alerts</h2>
            """
            
            # Add alerts to HTML
            for alert in alerts:
                alert_type = alert["type"].capitalize()
                alert_level = alert["level"]
                project = alert["project"]
                message = alert["message"]
                timestamp = alert["timestamp"]
                
                alerts_html += f"""
                <div class="alert alert-{alert_level}">
                    <div class="alert-title">{alert_type} Alert: {project}</div>
                    <div class="alert-message">{message}</div>
                    <div class="alert-meta">Generated: {timestamp}</div>
                </div>
                """
            
            alerts_html += """
                <div class="footer">
                    <p>Enhanced MLTracker - Alerts and Notifications System</p>
                </div>
            </body>
            </html>
            """
            
            # Write to file
            alerts_file = "alerts/project_alerts.html"
            with open(alerts_file, 'w') as f:
                f.write(alerts_html)
            
            print(f"✓ Generated {len(alerts)} alerts: {alerts_file}")
        else:
            print("No alerts generated based on current data")
        
        return alerts
    
    except Exception as e:
        print(f"Error generating alerts: {e}")
        return []

# Generate custom alerts
alerts = generate_custom_alerts(df_processed, df_with_anomalies)

# ----------------------
# SUMMARY AND CONCLUSIONS
# ----------------------
print("\n===== SUMMARY AND CONCLUSIONS =====")

print("\nKey findings from the Enhanced MLTracker analysis:")

# Deadline compliance rate
if 'Met Deadline?' in df_processed.columns:
    deadline_met_rate = df_processed['Met Deadline?'].eq('Yes').mean() * 100
    print(f"1. Project deadline compliance rate: {deadline_met_rate:.1f}%")

# Average delay
if 'Delay' in df_processed.columns:
    avg_delay = df_processed['Delay'].mean()
    print(f"2. Average project delay: {avg_delay:.1f} days")

# Most common reason for delay
if 'Reason Missed' in df_processed.columns:
    reason_counts = df_processed['Reason Missed'].value_counts()
    if not reason_counts.empty:
        most_common_reason = reason_counts.index[0]
        print(f"3. Most common reason for delay: {most_common_reason}")

# Predictive model performance
if model_results:
    best_model_name = max(model_results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model_accuracy = max(model_results.items(), key=lambda x: x[1]['accuracy'])[1]['accuracy'] * 100
    print(f"4. Best predictive model: {best_model_name.upper()} with accuracy {best_model_accuracy:.1f}%")

# Anomaly detection
anomaly_count = df_with_anomalies['is_anomaly'].sum() if 'is_anomaly' in df_with_anomalies.columns else 0
if anomaly_count > 0:
    print(f"5. Detected {anomaly_count} anomalous projects requiring attention")

# NLP insights
if nlp_results and 'topics' in nlp_results and nlp_results['topics']:
    top_topic_words = nlp_results['topics'][0]['words'][:3]
    print(f"6. Key terms in delay reasons: {', '.join(top_topic_words)}")

# Forecasting insights
if forecast_results and 'metrics' in forecast_results:
    trend = forecast_results['metrics'].get('trend_direction', '')
    print(f"7. Project completion trend: {trend}")

# Scenario simulation insights
if simulation_results and 'scenarios' in simulation_results:
    best_scenario = max(simulation_results['scenarios'].items(), key=lambda x: x[1]['impact'])
    print(f"8. Most positive scenario: '{best_scenario[0]}' (impact: {best_scenario[1]['impact']:+.2f})")

print("\nEnhanced features implemented:")
print("✓ Automated Report Generation")
print("✓ Interactive Dashboards")
print("✓ Anomaly Detection")
print("✓ Natural Language Processing for Delay Reasons")
print("✓ Advanced Forecasting with Prophet")
print("✓ Explainable AI (XAI) Enhancements")
print("✓ Real-Time Monitoring")
print("✓ Scenario Simulation")
print("✓ Custom Alerts and Notifications")

print("\nOutput files:")
print(f"- Automated Report: {report_file}")
print(f"- Interactive Dashboards: {', '.join(dashboard_files)}")
print("- Real-Time Monitoring: dashboards/real_time_monitoring.html")
print("- Alerts Dashboard: alerts/project_alerts.html")
print("- Charts: Multiple visualizations in the 'charts' directory")

print("\nEnhanced MLTracker analysis completed successfully!")

# ----------------------
# YEARLY UPDATE LINE CHART
# ----------------------
print("\nCreating a line chart for yearly updates...")

def plot_yearly_updates(df: pd.DataFrame) -> None:
    """
    Create a line chart showing yearly updates based on project completions.
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    """
    if 'Completion Date' not in df.columns:
        print("No 'Completion Date' column found for yearly updates.")
        return
    
    try:
        # Extract year from the Completion Date
        df['Completion Year'] = df['Completion Date'].dt.year
        
        # Group by year and count the number of completed projects
        yearly_data = df.groupby('Completion Year').size()
        
        # Plot the line chart
        plt.figure(figsize=(10, 6))
        plt.plot(yearly_data.index, yearly_data.values, marker='o', color=COLORS['primary'], linewidth=2)
        
        # Add labels and title
        plt.title('Yearly Project Completions', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Projects Completed', fontsize=14)
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the chart
        chart_path = 'charts/yearly_project_completions.png'
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f"✓ Created yearly updates line chart: {chart_path}")
    
    except Exception as e:
        print(f"Error creating yearly updates line chart: {e}")

# Call the function to create the chart
plot_yearly_updates(df_processed)