# MLTracker - Project Tracking Data Analysis
# Advanced ML-based project tracking and analysis system

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
import warnings
import networkx as nx
from networkx.algorithms import community
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap
# Removed unused import: seasonal_decompose
# Removed unused import: auto_arima
import os
import multiprocessing
from typing import Optional, Tuple, Dict, Any, List
# Removed unused import: ListedColormap

# Create charts directory if it doesn't exist
os.makedirs('charts', exist_ok=True)

# Try to import H2O (optional)
try:
    import h2o
    from h2o.automl import H2OAutoML
    h2o_available = True
except ImportError:
    h2o_available = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style for better visualizations
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("===== MLTracker - Project Tracking Data Analysis =====")
print("Initializing advanced project analytics system...\n")

# ----------------------
# 1. DATA COLLECTION & INGESTION
# ----------------------
print("1. DATA COLLECTION & INGESTION")

def fetch_data(source: str) -> pd.DataFrame:
    """
    Flexible data ingestion function that handles both URLs and local files
    
    Parameters:
    source (str): URL or file path to the project tracking data
    
    Returns:
    pandas.DataFrame: Loaded project tracking data
    """
    print("Fetching data from source...")
    
    if source.startswith(('http://', 'https://')):
        try:
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            print("Successfully retrieved data from URL")
            return pd.read_csv(StringIO(response.text))
        except Exception as e:
            print(f"Error fetching data from URL: {e}")
            print("Creating sample data for demonstration")
            return create_sample_data()
    else:
        try:
            print("Loading data from local file")
            return pd.read_csv(source)
        except Exception as e:
            print(f"Error loading local file: {e}")
            print("Creating sample data for demonstration")
            return create_sample_data()

def create_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Create sample project tracking data for demonstration
    
    Parameters:
    n_samples (int): Number of sample projects to create
    
    Returns:
    pandas.DataFrame: Sample project tracking data
    """
    np.random.seed(42)
    
    # Generate dates
    start_dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
    planned_durations = np.random.randint(10, 60, n_samples)
    deadlines = [start + pd.Timedelta(days=duration) for start, duration in zip(start_dates, planned_durations)]

    # Generate actual durations with some randomness
    actual_durations = [duration + np.random.randint(-10, 20) for duration in planned_durations]
    completion_dates = [start + pd.Timedelta(days=max(1, duration)) for start, duration in zip(start_dates, actual_durations)]

    # Generate deadline met status
    met_deadline = ['Yes' if comp <= dead else 'No' for comp, dead in zip(completion_dates, deadlines)]

    # Generate reasons for missed deadlines
    reasons = [
        'Resource constraints', 'Technical challenges', 'Scope changes', 
        'External dependencies', 'Poor estimation', 'Team changes'
    ]
    missed_reasons = [np.random.choice(reasons) if status == 'No' else None for status in met_deadline]

    # Create new deadlines for some projects
    new_deadlines = []
    for i, status in enumerate(met_deadline):
        if status == 'No' and np.random.random() < 0.7:
            new_deadlines.append(deadlines[i] + pd.Timedelta(days=np.random.randint(5, 20)))
        else:
            new_deadlines.append(None)

    # Create project names
    project_names = [f"Project_{i}" for i in range(n_samples)]

    # Create the dataframe
    df = pd.DataFrame({
        'Project Name': project_names,
        'Start Date': start_dates,
        'Deadline': deadlines,
        'Completion Date': completion_dates,
        'Met Deadline?': met_deadline,
        'Reason Missed': missed_reasons,
        'New Deadline': new_deadlines,
        'Days Taken': actual_durations
    })
    
    return df

# Try to load the dataset, fall back to sample data if file not found
try:
    file_path = "Project_Tracker_Fixed.csv"
    df = fetch_data(file_path)
except Exception as e:
    print(f"Error loading file: {e}")
    print("Falling back to sample data")
    df = create_sample_data()

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

def perform_eda(df: pd.DataFrame) -> None:
    """
    Comprehensive exploratory data analysis with visualizations
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    """
    print("Performing exploratory data analysis...")
    
    # Summary statistics
    print("\nSummary Statistics:")
    numeric_cols = ['Planned Duration', 'Actual Duration', 'Delay', 'Days Taken']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_cols:
        print(df[numeric_cols].describe())
    
    # Create visualizations one by one to avoid figure loop issues
    
    # 1. Distribution of project durations
    if 'Actual Duration' in df.columns:
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        sns.histplot(df['Actual Duration'].dropna(), kde=True, bins=20, ax=ax1)
        ax1.set_title('Distribution of Project Durations')
        ax1.set_xlabel('Duration (days)')
        ax1.set_ylabel('Frequency')
        fig1.savefig('charts/distribution_of_project_durations.png')
        plt.close(fig1)
        print("✓ Created project durations distribution chart")
    
    # 2. Deadline met vs missed
    if 'Met Deadline?' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        deadline_counts = df['Met Deadline?'].value_counts()
        if not deadline_counts.empty:
            deadline_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
            ax2.set_title('Projects Meeting Deadlines')
            ax2.set_ylabel('')
            fig2.savefig('charts/projects_meeting_deadlines.png')
            plt.close(fig2)
            print("✓ Created deadline compliance pie chart")
    
    # 3. Planned vs Actual Duration
    if 'Planned Duration' in df.columns and 'Actual Duration' in df.columns:
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        ax3.scatter(df['Planned Duration'], df['Actual Duration'], alpha=0.6)
        max_duration = max(df['Planned Duration'].max(), df['Actual Duration'].max())
        ax3.plot([0, max_duration], [0, max_duration], 'r--')
        ax3.set_title('Planned vs Actual Duration')
        ax3.set_xlabel('Planned Duration (days)')
        ax3.set_ylabel('Actual Duration (days)')
        ax3.grid(True)
        fig3.savefig('charts/planned_vs_actual_duration.png')
        plt.close(fig3)
        print("✓ Created planned vs actual duration chart")
    
    # 4. Reasons for missed deadlines
    if 'Reason Missed' in df.columns:
        reason_counts = df['Reason Missed'].value_counts().head(10)
        if not reason_counts.empty:
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            reason_counts.plot(kind='barh', ax=ax4)
            ax4.set_title('Top Reasons for Missed Deadlines')
            ax4.set_xlabel('Count')
            ax4.set_ylabel('Reason')
            fig4.savefig('charts/top_reasons_for_missed_deadlines.png')
            plt.close(fig4)
            print("✓ Created reasons for missed deadlines chart")
    
    # 5. Delay distribution
    if 'Delay' in df.columns:
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        sns.histplot(df['Delay'].dropna(), kde=True, bins=20, ax=ax5)
        ax5.axvline(x=0, color='r', linestyle='--')
        ax5.set_title('Distribution of Project Delays')
        ax5.set_xlabel('Delay (days)')
        ax5.set_ylabel('Frequency')
        fig5.savefig('charts/distribution_of_project_delays.png')
        plt.close(fig5)
        print("✓ Created delay distribution visualization")
    
    # 6. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        
        fig6, ax6 = plt.subplots(figsize=(14, 12))
        sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax6)
        ax6.set_title('Correlation Matrix')
        fig6.tight_layout()
        fig6.savefig('charts/correlation_matrix.png')
        plt.close(fig6)
        print("✓ Created correlation matrix visualization")
    
    # 7. Project completion by month
    if 'Completion Year' in df.columns and 'Completion Month' in df.columns:
        try:
            monthly_completion = df.groupby(['Completion Year', 'Completion Month']).size()
            monthly_completion.index = monthly_completion.index.map(lambda x: f"{x[0]}-{x[1]:02d}")
            
            if not monthly_completion.empty:
                fig7, ax7 = plt.subplots(figsize=(14, 7))
                monthly_completion.plot(kind='bar', ax=ax7)
                ax7.set_title('Project Completions by Month')
                ax7.set_xlabel('Year-Month')
                ax7.set_xticklabels(ax7.get_xticklabels(), rotation=90)
                fig7.tight_layout()
                fig7.savefig('charts/project_completions_by_month.png')
                plt.close(fig7)
                print("✓ Created monthly completions visualization")
        except Exception as e:
            print(f"Warning in monthly completions visualization: {e}")
    
    # 8. Deadline adherence by project complexity
    if 'Project Complexity' in df.columns and 'Deadline Met Binary' in df.columns:
        try:
            deadline_by_complexity = df.groupby('Project Complexity')['Deadline Met Binary'].mean() * 100
            
            if not deadline_by_complexity.empty:
                fig8, ax8 = plt.subplots(figsize=(10, 6))
                deadline_by_complexity.plot(kind='bar', ax=ax8)
                ax8.set_title('Deadline Adherence by Project Complexity')
                ax8.set_xlabel('Project Complexity')
                ax8.set_ylabel('Percentage of Deadlines Met')
                ax8.set_ylim(0, 100)
                fig8.savefig('charts/deadline_adherence_by_complexity.png')
                plt.close(fig8)
                print("✓ Created deadline adherence by complexity visualization")
        except Exception as e:
            print(f"Warning in deadline adherence visualization: {e}")
    
    # 9. Duration ratio boxplot
    if 'Project Complexity' in df.columns and 'Duration Ratio' in df.columns:
        try:
            fig9, ax9 = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='Project Complexity', y='Duration Ratio', data=df, ax=ax9)
            ax9.axhline(y=1, color='r', linestyle='--')
            ax9.set_title('Duration Ratio by Project Complexity')
            ax9.set_xlabel('Project Complexity')
            ax9.set_ylabel('Actual/Planned Duration Ratio')
            fig9.savefig('charts/duration_ratio_boxplot.png')
            plt.close(fig9)
            print("✓ Created duration ratio boxplot")
        except Exception as e:
            print(f"Warning in duration ratio boxplot: {e}")
    
    print("Exploratory data analysis completed")

# Perform EDA
try:
    perform_eda(df_processed)
except Exception as e:
    print(f"Error in EDA: {e}")
    print("Continuing with analysis...")

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
                ))
            
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

def optimize_model_with_optuna(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
    preprocessor: ColumnTransformer
) -> Optional[Dict[str, Any]]:
    """
    Optimize model hyperparameters using Optuna
    
    Parameters:
    X_train, X_test, y_train, y_test: Train and test data
    preprocessor: Data preprocessing pipeline
    
    Returns:
    dict: Optimized model and performance metrics
    """
    print("\nOptimizing model hyperparameters with Optuna...")
    
    try:
        # Set Optuna logging level to reduce verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Use a single study to avoid multiple parallel studies
        study_name = "xgboost_optimization"
        
        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0, log=True)
            }
            
            # Create model with trial parameters
            model = xgb.XGBClassifier(
                **params,
                random_state=42,
                eval_metric='logloss'
            )
            
            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Evaluate using cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            return cv_scores.mean()
        
        # Create and run Optuna study
        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(objective, n_trials=3)
        
        # Get best parameters
        best_params = study.best_params
        print(f"Best hyperparameters found: {best_params}")
        
        # Train model with best parameters
        best_model = xgb.XGBClassifier(
            **best_params,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Create pipeline
        best_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', best_model)
        ])
        
        # Train and evaluate
        best_pipeline.fit(X_train, y_train)
        y_pred = best_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"✓ Optimized XGBoost model - Accuracy: {accuracy:.4f}")
        
        return {
            'pipeline': best_pipeline,
            'accuracy': accuracy,
            'report': report,
            'best_params': best_params
        }
    except Exception as e:
        print(f"Error in Optuna optimization: {e}")
        return None

def train_h2o_automl(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
) -> Optional[Dict[str, Any]]:
    """
    Train models using H2O AutoML
    
    Parameters:
    X_train, X_test, y_train, y_test: Train and test data
    
    Returns:
    dict: H2O model performance metrics
    """
    if not h2o_available:
        print("Skipping H2O AutoML (not available)")
        return None
    
    print("\nTraining models with H2O AutoML...")
    
    try:
        # Check if Java is available
        import subprocess
        try:
            subprocess.check_call(['java', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error in H2O AutoML: Java is not available. Please install Java to use H2O.")
            return None
            
        # Initialize H2O
        h2o.init(nthreads=-1, max_mem_size="2G")
        
        # Combine features and target for training data
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # Convert to H2O frames
        train_h2o = h2o.H2OFrame(train_data)
        test_h2o = h2o.H2OFrame(test_data)
        
        # Identify features and target
        features = X_train.columns.tolist()
        target = y_train.name
        
        # Convert target to categorical
        train_h2o[target] = train_h2o[target].asfactor()
        test_h2o[target] = test_h2o[target].asfactor()
        
        # Run AutoML
        automl = H2OAutoML(max_models=3, seed=42, max_runtime_secs=60)
        automl.train(x=features, y=target, training_frame=train_h2o)
        
        # Get best model
        best_model = automl.leader
        
        # Evaluate on test data
        performance = best_model.model_performance(test_h2o)
        accuracy = 1 - performance.mean_per_class_error()
        
        print(f"✓ H2O AutoML completed - Best model: {best_model.model_id}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Shutdown H2O
        h2o.shutdown(prompt=False)
        
        return {
            'model_id': best_model.model_id,
            'accuracy': accuracy
        }
    
    except Exception as e:
        print(f"Error in H2O AutoML: {e}")
        try:
            h2o.shutdown(prompt=False)
        except Exception:
            pass
        return None

def perform_time_series_analysis_with_additional_insights(df: pd.DataFrame) -> None:
    """
    Perform advanced time series analysis with additional insights and visualizations.
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    """
    print("\nPerforming advanced time series analysis with additional insights...")
    
    try:
        # Prepare time series data
        if 'Completion Date' not in df.columns:
            print("Completion Date column not found for time series analysis")
            return
            
        # Create a time series of project completions by month
        df['Completion Month-Year'] = df['Completion Date'].dt.to_period('M')
        monthly_completions = df.groupby('Completion Month-Year').size()
        
        # Check if we have enough data
        if len(monthly_completions) < 6:
            print("Not enough time series data for analysis (need at least 6 months)")
            return
            
        # Convert to time series with proper datetime index
        ts = monthly_completions.to_timestamp().asfreq('M')
        
        # ----------------------
        # 1. Bar Chart of Monthly Completions
        # ----------------------
        fig1, ax1 = plt.subplots(figsize=(16, 8))
        
        # Plot the bar chart
        ts.plot(kind='bar', ax=ax1, color='skyblue', label='Monthly Completions')
        
        # Customize the plot
        ax1.set_title('Number of Projects Completed by Month (Bar Chart)', fontsize=16)
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Number of Projects', fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.legend(fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        fig1.tight_layout()
        fig1.savefig('charts/monthly_completions_bar_chart.png', dpi=300)
        plt.close(fig1)
        print("✓ Created bar chart of monthly completions")
        
        # ----------------------
        # 2. Cumulative Completions Line Chart
        # ----------------------
        cumulative_completions = ts.cumsum()
        
        fig2, ax2 = plt.subplots(figsize=(16, 8))
        
        # Plot the cumulative completions
        cumulative_completions.plot(
            ax=ax2, 
            marker='o', 
            markersize=8, 
            linewidth=2, 
            color='green',
            label='Cumulative Completions'
        )
        
        # Customize the plot
        ax2.set_title('Cumulative Project Completions Over Time', fontsize=16)
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('Cumulative Number of Projects', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        fig2.tight_layout()
        fig2.savefig('charts/cumulative_completions_line_chart.png', dpi=300)
        plt.close(fig2)
        print("✓ Created cumulative completions line chart")
        
        # ----------------------
        # 3. Heatmap of Project Completions by Month and Year
        # ----------------------
        try:
            # Extract year and month for heatmap
            df['Completion Year'] = df['Completion Date'].dt.year
            df['Completion Month'] = df['Completion Date'].dt.month_name()
            
            # Create pivot table for heatmap
            heatmap_data = df.pivot_table(
                index='Completion Month',
                columns='Completion Year',
                values='Project Name',
                aggfunc='count',
                fill_value=0
            )
            
            # Order months chronologically
            month_order = [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]
            heatmap_data = heatmap_data.reindex(month_order)
            
            # Create heatmap
            fig3, ax3 = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='g',
                cmap='YlGnBu',
                linewidths=0.5,
                ax=ax3
            )
            
            ax3.set_title('Monthly Project Completions Heatmap', fontsize=16)
            ax3.set_xlabel('Year', fontsize=12)
            ax3.set_ylabel('Month', fontsize=12)
            
            fig3.tight_layout()
            fig3.savefig('charts/monthly_completions_heatmap.png', dpi=300)
            plt.close(fig3)
            print("✓ Created monthly completions heatmap")
            
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            
    except Exception as e:
        print(f"Error in time series analysis: {e}")

def analyze_model_with_shap(
    model: Any, 
    X_test: pd.DataFrame, 
    preprocessor: ColumnTransformer
) -> Optional[shap.Explanation]:
    """
    Analyze model predictions using SHAP values
    
    Parameters:
    model: Trained model
    X_test: Test features
    preprocessor: Data preprocessing pipeline
    
    Returns:
    Optional[shap.Explanation]: SHAP values if successful, None otherwise
    """
    print("\nAnalyzing model predictions with SHAP...")
    
    try:
        # Transform test data
        X_test_processed = preprocessor.transform(X_test)
        
        # Check if SHAP values can be calculated for this model
        if not hasattr(model, 'predict') or not callable(getattr(model, 'predict')):
            print("Model doesn't support SHAP analysis")
            return None
        
        # Create explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test_processed)
        
        # Validate SHAP values
        if shap_values is None or len(shap_values) == 0:
            print("Could not generate valid SHAP values")
            return None
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_processed, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig('charts/shap_summary.png')
        plt.close()
        print("✓ SHAP summary plot created")
        
        # Create detailed SHAP plot for top samples
        if len(shap_values) > 0:
            plt.figure(figsize=(16, 10))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            plt.savefig('charts/shap_waterfall.png')
            plt.close()
            print("✓ SHAP waterfall plot created")
        
        return shap_values
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return None

def perform_network_analysis(df: pd.DataFrame) -> Tuple[Optional[nx.Graph], Optional[List[set]]]:
    """
    Perform network analysis to identify project dependencies and bottlenecks
    
    Parameters:
    df (pandas.DataFrame): Processed project tracking data
    
    Returns:
    tuple: (NetworkX graph, communities) if successful, (None, None) otherwise
    """
    print("\n5. NETWORK ANALYSIS FOR PROJECT DEPENDENCIES")
    print("Performing network analysis for project dependencies...")
    
    try:
        # Create a network based on project characteristics
        G = nx.Graph()
        
        # Add nodes (projects)
        if 'Project Name' not in df.columns:
            print("Project Name column not found for network analysis")
            return None, None
            
        for idx, row in df.iterrows():
            project_name = row['Project Name']
            if pd.notna(project_name):
                # Extract scalar values to avoid Series comparison issues
                deadline_met = row['Met Deadline?'] if 'Met Deadline?' in df.columns and pd.notna(row['Met Deadline?']) else 'Unknown'
                delay = float(row['Delay']) if 'Delay' in df.columns and pd.notna(row['Delay']) else 0
                complexity = row['Project Complexity'] if 'Project Complexity' in df.columns and pd.notna(row['Project Complexity']) else 'Unknown'
                
                G.add_node(
                    project_name, 
                    deadline_met=deadline_met,
                    delay=delay,
                    complexity=complexity
                )
        
        # Check if we have enough nodes
        if len(G.nodes) < 5:
            print("Not enough projects for meaningful network analysis")
            return None, None
        
        # Add edges based on similarity of project characteristics
        project_pairs = [(p1, p2) for p1 in G.nodes for p2 in G.nodes if p1 != p2]
        
        for p1, p2 in project_pairs:
            # Get node attributes
            attrs1 = G.nodes[p1]
            attrs2 = G.nodes[p2]
            
            # Calculate similarity score (weight for the edge)
            similarity = 0
            
            # Similarity based on complexity
            if attrs1['complexity'] == attrs2['complexity']:
                similarity += 0.3
            
            # Similarity based on deadline met status
            if attrs1['deadline_met'] == attrs2['deadline_met']:
                similarity += 0.3
            
            # Similarity based on delay (within 5 days)
            if abs(attrs1['delay'] - attrs2['delay']) <= 5:
                similarity += 0.4
            
            # Add edge if similarity is above threshold
            if similarity >= 0.7:
                G.add_edge(p1, p2, weight=similarity)
        
        # Calculate network metrics
        if len(G.nodes) > 0 and len(G.edges) > 0:
            print("\nNetwork Analysis Metrics:")
            print(f"Number of nodes (projects): {len(G.nodes)}")
            print(f"Number of edges (connections): {len(G.edges)}")
            
            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Find most central projects (potential bottlenecks)
            top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print("\nMost connected projects (potential dependencies):")
            for project, centrality in top_degree:
                print(f"  - {project}: {centrality:.4f}")
            
            print("\nKey bottleneck projects:")
            for project, centrality in top_betweenness:
                print(f"  - {project}: {centrality:.4f}")
            
            # Identify communities
            try:
                communities = community.greedy_modularity_communities(G)
                print(f"\nNumber of project clusters detected: {len(communities)}")
                
                # Create a custom colormap with distinct colors for each community
                colors = plt.cm.get_cmap('tab20', len(communities))
                
                # Create network visualization
                plt.figure(figsize=(16, 12))
                
                # Position nodes using spring layout
                pos = nx.spring_layout(G, k=0.3, iterations=50)
                
                # Draw edges first (light gray)
                nx.draw_networkx_edges(G, pos, alpha=0.1, width=1, edge_color='gray')
                
                # Draw nodes with community colors
                for i, comm in enumerate(communities):
                    nx.draw_networkx_nodes(
                        G, pos, nodelist=list(comm),
                        node_color=[colors(i)] * len(comm),
                        node_size=200,
                        alpha=0.8,
                        label=f'Cluster {i+1}'
                    )
                
                # Draw labels for nodes with high betweenness centrality
                high_betweenness = [n for n, v in betweenness_centrality.items() if v > np.quantile(list(betweenness_centrality.values()), 0.9)]
                labels = {n: n for n in high_betweenness}
                nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
                
                # Add legend and title
                plt.legend(scatterpoints=1, frameon=True, title='Project Clusters')
                plt.title('Project Dependency Network Analysis', fontsize=16)
                
                # Save the figure
                plt.tight_layout()
                plt.savefig('charts/project_dependency_network.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Project dependency network visualization created")
                
                # Analyze impact of delays
                delayed_projects = []
                for node, attrs in G.nodes(data=True):
                    # Check string value to avoid Series comparison
                    if isinstance(attrs.get('deadline_met'), str) and attrs.get('deadline_met') == 'No':
                        delayed_projects.append(node)
                
                # Find projects that might be affected by these delays
                affected_projects = set()
                for proj in delayed_projects:
                    affected_projects.update(G.neighbors(proj))
                
                affected_projects = affected_projects - set(delayed_projects)
                
                print(f"\nNumber of delayed projects: {len(delayed_projects)}")
                print(f"Number of potentially affected projects: {len(affected_projects)}")
                
                return G, communities
                
            except Exception as e:
                print(f"Error in community detection: {e}")
                return G, None
        else:
            print("Not enough connected projects for network analysis")
            return None, None
    except Exception as e:
        print(f"Error in network analysis: {e}")
        return None, None

# ----------------------
# MAIN ANALYSIS WORKFLOW
# ----------------------

if __name__ == "__main__":
    # This is needed to fix the multiprocessing issue with tsfresh
    multiprocessing.freeze_support()
    
    try:
        # Prepare data for machine learning
        X_train, X_test, y_train, y_test = prepare_data_for_ml(df_processed)

        # Build preprocessing pipeline
        preprocessor = build_preprocessing_pipeline(X_train)

        # Train classification models
        model_results = train_classification_models(X_train, X_test, y_train, y_test)

        # Optimize best model with Optuna
        optimized_model = optimize_model_with_optuna(X_train, X_test, y_train, y_test, preprocessor)

        # Train H2O AutoML models
        h2o_results = train_h2o_automl(X_train, X_test, y_train, y_test)

        # Perform time series analysis and forecasting
        perform_time_series_analysis_with_additional_insights(df_processed)

        # Analyze model with SHAP
        if model_results and 'xgb' in model_results:
            xgb_model = model_results['xgb']['pipeline'].named_steps['classifier']
            shap_values = analyze_model_with_shap(xgb_model, X_test, model_results['xgb']['pipeline'].named_steps['preprocessor'])

        # Perform network analysis
        G, communities = perform_network_analysis(df_processed)

        # ----------------------
        # SUMMARY AND CONCLUSIONS
        # ----------------------
        print("\n===== SUMMARY AND CONCLUSIONS =====")

        print("\nKey findings from the MLTracker analysis:")

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

        # Optimized model performance
        if optimized_model:
            print(f"5. Optimized XGBoost model accuracy: {optimized_model['accuracy'] * 100:.1f}%")

        # Network analysis insights
        if G and communities:
            print(f"6. Identified {len(communities)} distinct project clusters")
            print("7. Network analysis revealed key bottleneck projects that affect multiple other projects")

        print("\nRecommendations:")
        print("1. Implement buffer times in project planning based on historical data")
        print("2. Address the most common reasons for delays with targeted interventions")
        print("3. Consider seasonal patterns when scheduling project start dates")
        print("4. Use the predictive models to assess risk for new projects")
        print("5. Monitor projects with similar characteristics to those that historically missed deadlines")
        print("6. Pay special attention to bottleneck projects identified in the network analysis")
        print("7. Consider restructuring project dependencies to minimize cascade effects of delays")

        print("\nMLTracker analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in main analysis workflow: {e}")
        print("Analysis completed with errors.")