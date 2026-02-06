"""
Predictive Planning Hub - Powered by TabPFN & Databricks

This Streamlit app provides an interactive interface for supply chain planning
analytics using TabPFN, a foundation model for tabular data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from databricks import sql
from databricks.sdk.core import Config
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix,
)
import tabpfn_client
from tabpfn_client import TabPFNClassifier, TabPFNRegressor
import matplotlib.pyplot as plt
import mlflow
import os
from datetime import datetime

# TabPFN Time Series imports for forecasting
try:
    from tabpfn_time_series import (
        TimeSeriesDataFrame,
        FeatureTransformer,
        TabPFNTimeSeriesPredictor,
        TabPFNMode,
    )
    from tabpfn_time_series.data_preparation import generate_test_X
    from tabpfn_time_series.features import RunningIndexFeature, CalendarFeature, AutoSeasonalFeature
    TABPFN_TS_AVAILABLE = True
except ImportError:
    TABPFN_TS_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="Predictive Planning Hub | TabPFN + Databricks",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# ============================================================================
# LIGHT PURPLE THEME - PRIOR LABS INSPIRED
# ============================================================================
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global light purple background */
    .stApp {
        background: linear-gradient(180deg, #f8f7ff 0%, #f0eeff 50%, #ebe8ff 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Headers - dark text for readability */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
    }
    
    /* Body text */
    p, span, div {
        color: #2d2d44;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d5a 50%, #1a1a2e 100%);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 30%, rgba(102, 126, 234, 0.2) 0%, transparent 50%),
                    radial-gradient(circle at 80% 70%, rgba(167, 139, 250, 0.2) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #c4c4e0 !important;
        text-align: center;
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.7;
        position: relative;
        z-index: 1;
    }
    
    /* Logo container in hero */
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 2rem;
        margin-bottom: 2rem;
        position: relative;
        z-index: 1;
    }
    
    .logo-divider {
        color: #8888aa;
        font-size: 1.5rem;
        font-weight: 300;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    .metric-card-pink {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.25);
    }
    
    .metric-card-cyan {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.25);
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, #00b894 0%, #00d4aa 100%);
        box-shadow: 0 8px 32px rgba(0, 184, 148, 0.25);
    }
    
    .metric-number {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.95) !important;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Process cards - light with colored accents */
    .process-card {
        background: #ffffff;
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        height: 220px;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.08);
        transition: all 0.3s ease;
    }
    
    .process-card:hover {
        box-shadow: 0 8px 40px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    .process-card-demand { border-top: 4px solid #667eea; }
    .process-card-supply { border-top: 4px solid #f093fb; }
    .process-card-production { border-top: 4px solid #4facfe; }
    .process-card-distribution { border-top: 4px solid #00d4aa; }
    
    .process-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    
    .process-title-demand { color: #667eea !important; }
    .process-title-supply { color: #e056a0 !important; }
    .process-title-production { color: #4facfe !important; }
    .process-title-distribution { color: #00b894 !important; }
    
    .process-item {
        font-size: 0.9rem;
        color: #4a4a6a !important;
        margin: 0.5rem 0;
        padding-left: 0.5rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: #ffffff;
        border: 1px solid rgba(102, 126, 234, 0.12);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 12px rgba(102, 126, 234, 0.06);
    }
    
    .feature-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e !important;
        margin-bottom: 0.5rem;
    }
    
    .feature-text {
        font-size: 0.9rem;
        color: #5a5a7a !important;
        line-height: 1.5;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(167, 139, 250, 0.08) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .info-box-text {
        color: #2d2d44 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a2e !important;
        margin: 2rem 0 1.25rem 0;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .badge-evaluate { background: rgba(0, 184, 148, 0.15); color: #00a885 !important; }
    .badge-score { background: rgba(102, 126, 234, 0.15); color: #4a5fd9 !important; }
    .badge-classification { background: rgba(240, 147, 251, 0.15); color: #c044a0 !important; }
    .badge-regression { background: rgba(79, 172, 254, 0.15); color: #3498db !important; }
    .badge-forecast { background: rgba(255, 193, 7, 0.15); color: #f39c12 !important; }
    
    /* Powered by section */
    .powered-by {
        background: #ffffff;
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.08);
    }
    
    .powered-by-title {
        font-size: 0.85rem;
        color: #8888aa !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    
    .powered-by-name {
        font-size: 1.8rem;
        font-weight: 800;
        color: #667eea !important;
    }
    
    .powered-by-desc {
        color: #6a6a8a !important;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f7ff 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    section[data-testid="stSidebar"] .stRadio > label {
        color: #2d2d44 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
        color: #2d2d44 !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.2), transparent);
        margin: 2rem 0;
    }
    
    /* Status indicators */
    .status-success { color: #00b894 !important; font-weight: 500; }
    .status-warning { color: #fdcb6e !important; font-weight: 500; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6a6a8a !important;
        font-size: 0.85rem;
    }
    
    .footer a {
        color: #667eea !important;
        text-decoration: none;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Fix metric text colors */
    [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #4a4a6a !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff !important;
        border-radius: 10px !important;
        color: #2d2d44 !important;
    }
    
    /* Mode selector styling */
    .mode-card {
        background: #ffffff;
        border: 2px solid rgba(102, 126, 234, 0.15);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .mode-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
    }
    
    .mode-card-active {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Databricks configuration
cfg = Config()

# Dataset configurations
CATALOG = "tabpfn_databricks"
SCHEMA = "default"


@st.cache_resource(ttl=300, show_spinner="Connecting to Databricks...")
def get_connection(http_path: str):
    return sql.connect(
        server_hostname=cfg.host,
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
    )


@st.cache_data(ttl=600, show_spinner="Loading data...")
def load_table(_conn, table_name: str) -> pd.DataFrame:
    with _conn.cursor() as cursor:
        cursor.execute(f"SELECT * FROM {table_name}")
        return cursor.fetchall_arrow().to_pandas()


@st.cache_data(ttl=600, show_spinner="Fetching available tables...")
def get_available_tables(_conn, catalog: str, schema: str) -> list:
    """Get list of all tables in the schema."""
    with _conn.cursor() as cursor:
        cursor.execute(f"SHOW TABLES IN {catalog}.{schema}")
        result = cursor.fetchall()
        tables = [row[1] for row in result]  # tableName is typically the second column
        return sorted(tables)


def authenticate_tabpfn():
    token = os.environ.get("TABPFN_TOKEN")
    if token:
        tabpfn_client.set_access_token(token)
        return True
    return False


def prepare_features(df: pd.DataFrame, feature_cols: list, target_col: str = None):
    """Prepare features for modeling."""
    # Get categorical columns
    cat_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cat_cols:
        df_encoded = pd.get_dummies(df[feature_cols], columns=cat_cols, drop_first=True)
    else:
        df_encoded = df[feature_cols].copy()
    
    X = df_encoded.values
    y = df[target_col].values if target_col else None
    
    return X, y, df_encoded.columns.tolist()


def run_classification(X_train, y_train, X_test=None, y_test=None):
    """Run classification with TabPFN."""
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)
    
    if X_test is not None:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        
        results = {
            "predictions": y_pred,
            "probabilities": y_pred_proba,
            "model": clf
        }
        
        if y_test is not None:
            accuracy = accuracy_score(y_test, y_pred)
            n_classes = len(np.unique(y_train))
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
                except ValueError:
                    roc_auc = None
            results["accuracy"] = accuracy
            results["roc_auc"] = roc_auc
            results["y_test"] = y_test
        
        return results
    return {"model": clf}


def run_regression(X_train, y_train, X_test=None, y_test=None):
    """Run regression with TabPFN."""
    reg = TabPFNRegressor()
    reg.fit(X_train, y_train)
    
    if X_test is not None:
        y_pred = reg.predict(X_test)
        
        results = {
            "predictions": y_pred,
            "model": reg
        }
        
        if y_test is not None:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results["rmse"] = rmse
            results["mae"] = mae
            results["r2"] = r2
            results["y_test"] = y_test
        
        return results
    return {"model": reg}


def pandas_to_time_series_dataframe(df, item_id_col, timestamp_col, target_col):
    """
    Convert a pandas DataFrame to a TimeSeriesDataFrame.
    
    Args:
        df: pandas DataFrame with time series data
        item_id_col: Column name for series identifier
        timestamp_col: Column name for timestamps
        target_col: Column name for target values
    
    Returns:
        TimeSeriesDataFrame
    """
    if not TABPFN_TS_AVAILABLE:
        raise ImportError("tabpfn-time-series package is required for forecasting")
    
    # Prepare the data with proper index
    df_ts = df[[item_id_col, timestamp_col, target_col]].copy()
    df_ts = df_ts.rename(columns={
        item_id_col: 'item_id',
        timestamp_col: 'timestamp',
        target_col: 'target'
    })
    
    # Ensure timestamp is datetime64[ns] dtype without timezone (required by TimeSeriesDataFrame)
    df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp']).dt.tz_localize(None)
    # Force to datetime64[ns] dtype
    df_ts['timestamp'] = df_ts['timestamp'].astype('datetime64[ns]')
    
    df_ts = df_ts.sort_values(['item_id', 'timestamp'])
    df_ts = df_ts.set_index(['item_id', 'timestamp'])
    
    return TimeSeriesDataFrame(df_ts)


def create_lag_features(series: np.ndarray, n_lags: int = 12):
    """Create lag features for time series forecasting (legacy fallback)."""
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def add_calendar_features(X: np.ndarray, dates, n_lags: int):
    """Add calendar features to lag features (legacy fallback)."""
    dates_subset = pd.to_datetime(dates[n_lags:])
    months = np.array([d.month for d in dates_subset])
    years = np.array([d.year for d in dates_subset])
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    return np.column_stack([X, month_sin, month_cos, years - years.min()])


def run_forecasting_tabpfn_ts(tsdf_train, tsdf_test_X, tsdf_test_y=None, forecast_horizon=6):
    """
    Run time series forecasting with TabPFNTimeSeriesPredictor.
    
    Args:
        tsdf_train: TimeSeriesDataFrame for training
        tsdf_test_X: TimeSeriesDataFrame with test features
        tsdf_test_y: Optional TimeSeriesDataFrame with actual test values
        forecast_horizon: Number of periods to forecast
    
    Returns:
        dict with predictions, metrics, and prediction intervals
    """
    if not TABPFN_TS_AVAILABLE:
        raise ImportError("tabpfn-time-series package is required for forecasting")
    
    # Add temporal and seasonal features
    features = [
        RunningIndexFeature(),
        CalendarFeature(),
        AutoSeasonalFeature()
    ]
    transformer = FeatureTransformer(features)
    train_feat, test_X_feat = transformer.transform(tsdf_train, tsdf_test_X)
    
    # Initialize predictor
    predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.CLIENT)
    
    # Generate forecasts
    pred = predictor.predict(train_feat, test_X_feat)
    
    # Extract predictions
    pred_col = 'mean' if 'mean' in pred.columns else pred.columns[0]
    y_pred = pred[pred_col].values
    
    results = {
        "predictions": y_pred,
        "predictor": predictor,
        "features_used": list(train_feat.columns),
    }
    
    # Extract prediction intervals if available
    if 'q0.1' in pred.columns and 'q0.9' in pred.columns:
        results["y_lower"] = pred['q0.1'].values
        results["y_upper"] = pred['q0.9'].values
    elif 'q0.05' in pred.columns and 'q0.95' in pred.columns:
        results["y_lower"] = pred['q0.05'].values
        results["y_upper"] = pred['q0.95'].values
    else:
        # Simple estimate if quantiles not available
        results["y_lower"] = y_pred * 0.9
        results["y_upper"] = y_pred * 1.1
    
    # Calculate metrics if actual values provided
    if tsdf_test_y is not None:
        y_test = tsdf_test_y['target'].values
        min_len = min(len(y_test), len(y_pred))
        y_test = y_test[:min_len]
        y_pred_eval = y_pred[:min_len]
        
        mae = mean_absolute_error(y_test, y_pred_eval)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_eval))
        mape = np.mean(np.abs((y_test - y_pred_eval) / (y_test + 1e-8))) * 100
        
        results["mae"] = mae
        results["rmse"] = rmse
        results["mape"] = mape
        results["y_test"] = y_test
        
        # Calculate coverage
        y_lower = results["y_lower"][:min_len]
        y_upper = results["y_upper"][:min_len]
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
        results["coverage"] = coverage
    
    return results


def run_forecasting(X_train, y_train, X_test=None, y_test=None):
    """Run time series forecasting with TabPFN (legacy fallback)."""
    reg = TabPFNRegressor()
    reg.fit(X_train, y_train)
    
    if X_test is not None:
        y_pred = reg.predict(X_test)
        
        results = {
            "predictions": y_pred,
            "model": reg
        }
        
        # Try to get prediction intervals
        try:
            y_lower = reg.predict(X_test, output_type="quantiles", quantiles=[0.1]).flatten()
            y_upper = reg.predict(X_test, output_type="quantiles", quantiles=[0.9]).flatten()
            results["y_lower"] = y_lower
            results["y_upper"] = y_upper
        except Exception:
            results["y_lower"] = None
            results["y_upper"] = None
        
        if y_test is not None:
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
            results["mae"] = mae
            results["rmse"] = rmse
            results["mape"] = mape
            results["y_test"] = y_test
            
            if results["y_lower"] is not None:
                coverage = np.mean((y_test >= results["y_lower"]) & (y_test <= results["y_upper"]))
                results["coverage"] = coverage
        
        return results
    return {"model": reg}


# Environment variables
http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
tabpfn_token = os.environ.get("TABPFN_TOKEN", "")


def setup_mlflow_experiment(conn):
    """Set up MLflow experiment using the same naming convention as notebooks."""
    error_details = []
    
    try:
        # Set MLflow tracking URI to Databricks workspace
        mlflow.set_tracking_uri("databricks")
        
        # Get current user from Databricks SQL
        cursor = conn.cursor()
        cursor.execute("SELECT current_user()")
        current_user = cursor.fetchone()[0]
        cursor.close()
        
        experiment_name = f"/Users/{current_user}/tabpfn-databricks"
        
        # Try to get or create the experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                experiment = mlflow.get_experiment(experiment_id)
            else:
                experiment_id = experiment.experiment_id
        except Exception as exp_error:
            # If get/create fails, try set_experiment as fallback
            error_details.append(f"get/create experiment: {exp_error}")
            experiment = mlflow.set_experiment(experiment_name)
            experiment_id = experiment.experiment_id
        
        # Set the experiment as active
        mlflow.set_experiment(experiment_name)
        
        # Get workspace host for constructing URLs
        workspace_host = cfg.host
        
        return {
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
            "workspace_host": workspace_host,
            "current_user": current_user,
        }, None
    except Exception as e:
        error_details.append(f"primary setup: {e}")
        # Fallback experiment name
        experiment_name = "/Shared/tabpfn-databricks-app"
        try:
            mlflow.set_tracking_uri("databricks")
            experiment = mlflow.set_experiment(experiment_name)
            experiment_id = experiment.experiment_id
            workspace_host = cfg.host
            return {
                "experiment_name": experiment_name,
                "experiment_id": experiment_id,
                "workspace_host": workspace_host,
                "current_user": "shared",
                "fallback": True,
                "error_details": "; ".join(error_details),
            }, None
        except Exception as e2:
            error_details.append(f"fallback setup: {e2}")
            return None, "; ".join(error_details)


def log_to_mlflow(run_name, params, metrics, task_type, operation_mode, mlflow_config=None):
    """Log evaluation or scoring run to MLflow."""
    try:
        # Ensure we're using the correct experiment
        if mlflow_config and mlflow_config.get('experiment_name'):
            mlflow.set_experiment(mlflow_config['experiment_name'])
        
        with mlflow.start_run(run_name=run_name):
            # Log common parameters
            mlflow.log_param("source", "streamlit_app")
            mlflow.log_param("task_type", task_type)
            mlflow.log_param("operation_mode", operation_mode)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            # Log custom parameters
            for key, value in params.items():
                try:
                    mlflow.log_param(key, value)
                except Exception:
                    pass
            
            # Log metrics
            for key, value in metrics.items():
                try:
                    if value is not None and not np.isnan(value):
                        mlflow.log_metric(key, float(value))
                except Exception:
                    pass
            
            run_id = mlflow.active_run().info.run_id
            
            # Construct run URL if config is available
            run_url = None
            if mlflow_config:
                workspace_host = mlflow_config.get("workspace_host", "")
                experiment_id = mlflow_config.get("experiment_id", "")
                if workspace_host and experiment_id:
                    # Remove protocol if already present in host
                    if workspace_host.startswith("https://"):
                        base_url = workspace_host
                    elif workspace_host.startswith("http://"):
                        base_url = workspace_host
                    else:
                        base_url = f"https://{workspace_host}"
                    run_url = f"{base_url}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
            
            return run_id, run_url, None
    except Exception as e:
        return None, None, str(e)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    # Databricks logo as inline SVG
    databricks_icon = '''<svg width="50" height="50" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path d="M50 10L90 30V50L50 70L10 50V30L50 10Z" fill="#FF3621"/>
        <path d="M50 30L90 50V70L50 90L10 70V50L50 30Z" fill="#FF3621" opacity="0.7"/>
        <path d="M50 10L90 30L50 50L10 30L50 10Z" fill="#FF6B4A"/>
    </svg>'''
    
    # Prior Labs logo as inline SVG (stylized "P" with text)
    prior_labs_icon = '''<svg width="120" height="40" viewBox="0 0 120 40" xmlns="http://www.w3.org/2000/svg">
        <rect x="2" y="5" width="30" height="30" rx="6" fill="#667eea"/>
        <text x="10" y="28" font-family="Inter, Arial, sans-serif" font-size="20" font-weight="700" fill="white">P</text>
        <text x="38" y="26" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="600" fill="#1a1a2e">Prior Labs</text>
    </svg>'''
    
    st.markdown(f"""
    <div style="padding: 1rem 0; text-align: center;">
        <div style="margin-bottom: 0.5rem;">
            {databricks_icon}
            <div style="font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 600; color: #1a1a2e; letter-spacing: -0.5px;">databricks</div>
        </div>
        <div style="color: #8888aa; font-size: 1.5rem; font-weight: 300; margin: 0.5rem 0;">×</div>
        <div style="margin-top: 0.5rem;">
            {prior_labs_icon}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Navigation - use session state to determine initial selection
    nav_options = ["🏠 Home", "⚡ Predictions"]
    current_index = 0 if st.session_state.current_page == "home" else 1
    page = st.radio("Navigation", nav_options, index=current_index, label_visibility="collapsed")
    st.session_state.current_page = "home" if page == "🏠 Home" else "predictions"
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Status
    st.markdown("**Connection Status**")
    if http_path and not http_path.startswith("YOUR_"):
        st.markdown('<span class="status-success">✓ SQL Warehouse Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warning">○ SQL Warehouse</span>', unsafe_allow_html=True)
    
    if tabpfn_token and not tabpfn_token.startswith("YOUR_"):
        st.markdown('<span class="status-success">✓ TabPFN API Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warning">○ TabPFN API</span>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 0.75rem; color: #6a6a8a; text-align: center;">
        <a href="https://priorlabs.ai/" target="_blank" style="color: #667eea;">Prior Labs</a> • 
        <a href="https://databricks.com/" target="_blank" style="color: #ff3621;">Databricks</a>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# HOME PAGE
# ============================================================================
if st.session_state.current_page == "home":
    
    # Prior Labs logo SVG for hero (white version)
    prior_labs_hero = '''<svg width="140" height="45" viewBox="0 0 140 45" xmlns="http://www.w3.org/2000/svg">
        <rect x="2" y="5" width="35" height="35" rx="7" fill="#667eea"/>
        <text x="11" y="32" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="700" fill="white">P</text>
        <text x="44" y="30" font-family="Inter, Arial, sans-serif" font-size="18" font-weight="600" fill="white">Prior Labs</text>
    </svg>'''
    
    # Hero Section
    st.markdown(f"""
    <div class="hero-container">
        <div class="logo-container">
            <img src="https://www.databricks.com/wp-content/uploads/2022/06/db-nav-logo.svg" width="150" alt="Databricks">
            <span class="logo-divider">×</span>
            {prior_labs_hero}
        </div>
        <h1 class="hero-title">Predictive Planning Hub</h1>
        <p class="hero-subtitle">
            Your centralized platform for predictive analytics across the entire planning value chain. 
            Powered by TabPFN — the foundation model for tabular data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-number">1</div><div class="metric-label">Foundation Model</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card metric-card-green"><div class="metric-number">∞</div><div class="metric-label">Use Cases</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How it works section
    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="feature-card"><div class="feature-title">1️⃣ Select Dataset</div><div class="feature-text">Browse all available tables in your schema and select one for prediction.</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><div class="feature-title">2️⃣ Choose Mode</div><div class="feature-text">Select prediction type: Classification, Regression, or Time Series Forecast.</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><div class="feature-title">3️⃣ Configure</div><div class="feature-text">Specify feature columns and target/label column for your model.</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="feature-card"><div class="feature-title">4️⃣ Execute</div><div class="feature-text">Run evaluation (with metrics) or scoring (predictions only).</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Operation Modes
    st.markdown('<div class="section-header">Operation Modes</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card" style="border-left: 4px solid #00b894;">
            <div class="feature-title">📊 Evaluate Mode</div>
            <div class="feature-text">
                <strong>Uses: _train datasets</strong><br><br>
                Train on a portion of the data, predict on the rest, and evaluate performance against known labels. 
                Get accuracy, ROC AUC, RMSE, R², and other metrics to validate model performance.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card" style="border-left: 4px solid #667eea;">
            <div class="feature-title">🎯 Score Mode</div>
            <div class="feature-text">
                <strong>Uses: _score datasets</strong><br><br>
                Train on the full _train dataset, then generate predictions on the _score dataset. 
                Use this for production scoring where you need predictions on new, unseen data.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="info-box"><div class="info-box-text"><strong>Data Split Convention:</strong> Datasets are pre-split into <code>*_train</code> (80%) and <code>*_score</code> (20%) tables. Evaluate mode uses only the _train table with internal cross-validation. Score mode trains on _train and predicts on _score.</div></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Powered by TabPFN
    st.markdown('<div class="section-header">Powered by TabPFN</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **TabPFN** (Tabular Prior-Data Fitted Network) is a foundation model for tabular data 
        developed by [Prior Labs](https://priorlabs.ai/). It has been pretrained on millions of 
        synthetic datasets, enabling it to make accurate predictions on new data **without any training**.
        
        **Key Benefits:**
        - ✅ **Zero training time** — Predictions in seconds
        - ✅ **No hyperparameter tuning** — Works out of the box  
        - ✅ **Uncertainty quantification** — Built-in prediction intervals
        - ✅ **Strong performance** — Competitive with tuned XGBoost, Random Forest
        - ✅ **Published in Nature** — Rigorous scientific validation
        """)
    with col2:
        prior_labs_powered = '''<svg width="140" height="45" viewBox="0 0 140 45" xmlns="http://www.w3.org/2000/svg">
            <rect x="2" y="5" width="35" height="35" rx="7" fill="#667eea"/>
            <text x="11" y="32" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="700" fill="white">P</text>
            <text x="44" y="30" font-family="Inter, Arial, sans-serif" font-size="18" font-weight="600" fill="#1a1a2e">Prior Labs</text>
        </svg>'''
        st.markdown(f'<div class="powered-by"><div class="powered-by-title">Powered by</div><div style="margin: 0.75rem 0;">{prior_labs_powered}</div><div class="powered-by-desc">Foundation Model for Tabular Data</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    cta1, cta2, cta3 = st.columns([1, 2, 1])
    with cta2:
        if st.button("⚡ Start Making Predictions", type="primary", use_container_width=True):
            st.session_state.current_page = "predictions"
            st.rerun()
    
    st.markdown('<div class="footer">Built with <a href="https://databricks.com">Databricks</a> and <a href="https://priorlabs.ai">TabPFN by Prior Labs</a></div>', unsafe_allow_html=True)

# ============================================================================
# PREDICTIONS PAGE
# ============================================================================
else:
    st.markdown('<div class="section-header" style="margin-top: 0;">⚡ Predictions</div>', unsafe_allow_html=True)
    
    # Check connections
    if not http_path or http_path.startswith("YOUR_"):
        st.error("⚠️ SQL Warehouse not configured. Edit `app.yaml` to set `DATABRICKS_HTTP_PATH`.")
        st.stop()

    if not authenticate_tabpfn():
        st.error("⚠️ TabPFN token not configured. Edit `app.yaml` to set `TABPFN_TOKEN`.")
        st.stop()
    
    try:
        conn = get_connection(http_path)
        
        # Set up MLflow experiment (same as notebooks)
        mlflow_config, mlflow_error = setup_mlflow_experiment(conn)
        if mlflow_config:
            st.sidebar.markdown(f"**MLflow Experiment:**")
            st.sidebar.code(mlflow_config['experiment_name'], language=None)
            if mlflow_config.get('fallback'):
                st.sidebar.warning(f"Using fallback experiment. Error: {mlflow_config.get('error_details', 'unknown')}")
        elif mlflow_error:
            st.sidebar.warning(f"MLflow setup failed: {mlflow_error}")
        
        # ====================================================================
        # STEP 1: Select Dataset
        # ====================================================================
        st.markdown("### 1️⃣ Select Dataset")
        
        # Get available tables
        all_tables = get_available_tables(conn, CATALOG, SCHEMA)
        
        # Filter to show only base table names (without _train/_score suffix for display)
        base_tables = set()
        for t in all_tables:
            if t.endswith("_train"):
                base_tables.add(t[:-6])  # Remove _train suffix
            elif t.endswith("_score"):
                base_tables.add(t[:-6])  # Remove _score suffix
            else:
                base_tables.add(t)
        
        base_tables = sorted(list(base_tables))
        
        if not base_tables:
            st.warning("No tables found in the schema. Please run the data preparation notebook first.")
            st.stop()
        
        selected_base_table = st.selectbox(
            "Select a dataset",
            options=base_tables,
            help="Choose a dataset for prediction. The system will automatically use _train for evaluation and _score for scoring."
        )
        
        # Check if train/score versions exist
        train_table = f"{selected_base_table}_train"
        score_table = f"{selected_base_table}_score"
        has_train = train_table in all_tables
        has_score = score_table in all_tables
        
        if has_train:
            st.success(f"✓ Found `{train_table}` and `{score_table}` tables")
        else:
            st.warning(f"⚠️ No _train/_score split found. Using `{selected_base_table}` directly.")
            train_table = selected_base_table
            score_table = selected_base_table
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # ====================================================================
        # STEP 2: Choose Mode
        # ====================================================================
        st.markdown("### 2️⃣ Choose Prediction Mode")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            prediction_mode = st.radio(
                "Prediction Type",
                ["Classification", "Regression", "Forecast"],
                help="Classification for categorical targets, Regression for continuous targets, Forecast for time series"
            )
        
        with col2:
            operation_mode = st.radio(
                "Operation Mode",
                ["Evaluate", "Score"],
                help="Evaluate: test model on _train data with metrics. Score: generate predictions on _score data."
            )
        
        with col3:
            st.markdown(f"""
            <div style="padding: 1rem; background: rgba(102, 126, 234, 0.05); border-radius: 10px; margin-top: 0.5rem;">
                <strong>Selected:</strong><br>
                <span class="badge badge-{prediction_mode.lower()}">{prediction_mode}</span>
                <span class="badge badge-{operation_mode.lower()}">{operation_mode}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # ====================================================================
        # STEP 3: Load Data & Configure Columns
        # ====================================================================
        st.markdown("### 3️⃣ Configure Features & Target")
        
        # Load the training data for column selection
        with st.spinner(f"Loading {train_table}..."):
            df_train = load_table(conn, f"{CATALOG}.{SCHEMA}.{train_table}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Training Rows", f"{len(df_train):,}")
        with col2:
            st.metric("Columns", f"{len(df_train.columns)}")
        
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df_train.head(20), use_container_width=True)
        
        # Column configuration
        all_columns = df_train.columns.tolist()
        
        # Identify numeric and categorical columns
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df_train.select_dtypes(include=['datetime64']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target column selection
            target_col = st.selectbox(
                "Target/Label Column",
                options=all_columns,
                index=len(all_columns) - 1 if all_columns else 0,
                help="The column you want to predict"
            )
            
            if prediction_mode == "Forecast":
                # Additional columns for time series
                date_col = st.selectbox(
                    "Date Column",
                    options=datetime_cols + [c for c in all_columns if 'date' in c.lower()],
                    help="Column containing timestamps"
                )
                
                series_id_col = st.selectbox(
                    "Series ID Column (optional)",
                    options=["None"] + categorical_cols,
                    help="Column identifying different time series (e.g., product_id, region)"
                )
                if series_id_col == "None":
                    series_id_col = None
        
        with col2:
            # Feature columns selection
            default_features = [c for c in all_columns if c != target_col]
            
            if prediction_mode == "Forecast":
                # For forecasting, exclude date and series_id from features
                exclude_from_features = [target_col, date_col]
                if series_id_col:
                    exclude_from_features.append(series_id_col)
                default_features = [c for c in all_columns if c not in exclude_from_features]
            
            feature_cols = st.multiselect(
                "Feature Columns",
                options=[c for c in all_columns if c != target_col],
                default=default_features[:min(10, len(default_features))],  # Limit default selection
                help="Columns to use as input features"
            )
        
        if not feature_cols:
            st.warning("Please select at least one feature column.")
            st.stop()
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # ====================================================================
        # STEP 4: Model Configuration
        # ====================================================================
        st.markdown("### 4️⃣ Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            random_state = st.number_input("Random Seed", 0, 9999, 42)
        
        with col2:
            if operation_mode == "Evaluate":
                test_size = st.slider("Test Set Size (%)", 10, 50, 20)
            else:
                test_size = None
        
        with col3:
            if len(df_train) > 3000:
                max_samples = st.slider("Max Training Samples", 500, min(5000, len(df_train)), 2000)
            else:
                max_samples = None
        
        if prediction_mode == "Forecast":
            st.markdown("""
            <div style="padding: 0.75rem; background: rgba(102, 126, 234, 0.08); border-radius: 8px; margin-bottom: 1rem;">
                <strong>TabPFN Time Series</strong><br>
                <span style="font-size: 0.9rem; color: #5a5a7a;">
                    Uses automatic feature engineering with RunningIndexFeature, CalendarFeature, and AutoSeasonalFeature.
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            forecast_horizon = st.slider("Forecast Horizon", 1, 24, 6, help="Number of periods to forecast ahead")
            
            if series_id_col:
                unique_series = df_train[series_id_col].unique().tolist()
                selected_series = st.selectbox("Select Time Series", options=unique_series)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # ====================================================================
        # RUN PREDICTION
        # ====================================================================
        button_label = f"⚡ Run {operation_mode}" if operation_mode == "Evaluate" else "⚡ Generate Predictions"
        
        if st.button(button_label, type="primary", use_container_width=True):
            with st.spinner(f"Running TabPFN {prediction_mode.lower()}..."):
                
                # ============================================================
                # FORECASTING
                # ============================================================
                if prediction_mode == "Forecast":
                    # Check if TabPFN Time Series is available
                    if not TABPFN_TS_AVAILABLE:
                        st.error("⚠️ tabpfn-time-series package is not installed. Please install it to use forecasting features.")
                        st.stop()
                    
                    # Prepare time series data using TimeSeriesDataFrame
                    if series_id_col:
                        df_series = df_train[df_train[series_id_col] == selected_series].sort_values(date_col).reset_index(drop=True)
                        # Create a temporary series_id column for single series
                        df_series['_series_id'] = selected_series
                    else:
                        df_series = df_train.sort_values(date_col).reset_index(drop=True)
                        df_series['_series_id'] = 'single_series'
                    
                    if len(df_series) < forecast_horizon + 10:
                        st.error("Not enough data points for the selected configuration.")
                        st.stop()
                    
                    # Convert to TimeSeriesDataFrame
                    tsdf = pandas_to_time_series_dataframe(
                        df_series,
                        item_id_col='_series_id',
                        timestamp_col=date_col,
                        target_col=target_col
                    )
                    
                    if operation_mode == "Evaluate":
                        # Split into training and test portions
                        train_tsdf, test_tsdf = tsdf.train_test_split(prediction_length=forecast_horizon)
                        test_X = generate_test_X(train_tsdf, forecast_horizon)
                        
                        # Get dates for visualization
                        train_dates = train_tsdf.index.get_level_values('timestamp')
                        # Note: test_tsdf from train_test_split contains the full series, not just test portion
                        # Get only the last forecast_horizon dates from the original series
                        test_dates = tsdf.index.get_level_values('timestamp')[-forecast_horizon:]
                        train_values = train_tsdf['target'].values
                        
                        # Get actual test values (last forecast_horizon values from original series)
                        y_test_actual = tsdf['target'].values[-forecast_horizon:]
                        
                        # Run forecasting with TabPFNTimeSeriesPredictor
                        # Pass None for test_tsdf since we'll compute metrics ourselves
                        results = run_forecasting_tabpfn_ts(
                            train_tsdf, test_X, None, forecast_horizon
                        )
                        
                        # Compute metrics
                        y_pred = results['predictions']
                        min_len = min(len(y_test_actual), len(y_pred))
                        y_test_eval = y_test_actual[:min_len]
                        y_pred_eval = y_pred[:min_len]
                        
                        mae = mean_absolute_error(y_test_eval, y_pred_eval)
                        rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))
                        mape = np.mean(np.abs((y_test_eval - y_pred_eval) / (y_test_eval + 1e-8))) * 100
                        
                        # Calculate coverage
                        y_lower = results['y_lower'][:min_len]
                        y_upper = results['y_upper'][:min_len]
                        coverage = np.mean((y_test_eval >= y_lower) & (y_test_eval <= y_upper))
                        
                        # Display results
                        st.markdown('<div class="section-header">📊 Forecast Evaluation Results</div>', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("MAE", f"{mae:,.2f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:,.2f}")
                        with col3:
                            st.metric("MAPE", f"{mape:.1f}%")
                        with col4:
                            st.metric("80% Coverage", f"{coverage:.0%}")
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(12, 5))
                        fig.patch.set_facecolor('#f8f7ff')
                        ax.set_facecolor('#f8f7ff')
                        ax.plot(train_dates, train_values, color='#667eea', linewidth=1.5, label='Training', alpha=0.7)
                        ax.plot(test_dates, y_test_eval, color='#00b894', linewidth=2, marker='o', markersize=6, label='Actual')
                        ax.plot(test_dates, y_pred_eval, color='#e056a0', linewidth=2, marker='s', markersize=6, linestyle='--', label='Forecast')
                        if results.get('y_lower') is not None:
                            ax.fill_between(test_dates, y_lower, y_upper, alpha=0.15, color='#e056a0', label='80% Interval')
                        ax.set_xlabel('Date', color='#2d2d44')
                        ax.set_ylabel(target_col, color='#2d2d44')
                        ax.set_title(f'Forecast Evaluation (TabPFNTimeSeriesPredictor)', color='#1a1a2e', fontweight='bold')
                        ax.legend(facecolor='#ffffff', edgecolor='#e0e0e0', labelcolor='#2d2d44')
                        ax.tick_params(colors='#4a4a6a')
                        ax.grid(True, alpha=0.3, color='#667eea')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Results table
                        results_df = pd.DataFrame({
                            "Date": pd.to_datetime(test_dates).strftime('%Y-%m-%d'),
                            "Actual": y_test_eval,
                            "Forecast": y_pred_eval,
                            "Error": y_test_eval - y_pred_eval
                        })
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Log to MLflow
                        run_id, run_url, mlflow_err = log_to_mlflow(
                            run_name=f"app_{selected_base_table}_forecast_eval",
                            params={
                                "dataset": selected_base_table,
                                "model_type": "TabPFNTimeSeriesPredictor",
                                "tabpfn_mode": "CLIENT",
                                "task": "time_series_forecasting",
                                "target_col": target_col,
                                "date_col": date_col,
                                "series_id": selected_series if series_id_col else "single_series",
                                "forecast_horizon": forecast_horizon,
                                "train_samples": len(train_values),
                                "test_samples": len(y_test_eval),
                                "features_used": str(results.get('features_used', [])),
                                "random_state": random_state,
                            },
                            metrics={
                                "mae": mae,
                                "rmse": rmse,
                                "mape": mape,
                            },
                            task_type="forecast",
                            operation_mode="evaluate",
                            mlflow_config=mlflow_config
                        )
                        if run_id:
                            if run_url:
                                st.write(f"📋 MLflow Run: {run_id}")
                                st.link_button("View in MLflow", run_url)
                            else:
                                st.caption(f"📋 MLflow Run ID: `{run_id}`")
                        elif mlflow_err:
                            st.caption(f"⚠️ MLflow logging failed: {mlflow_err}")
                    
                    else:  # Score mode
                        # Load score data
                        with st.spinner(f"Loading {score_table}..."):
                            df_score = load_table(conn, f"{CATALOG}.{SCHEMA}.{score_table}")
                        
                        # For time series scoring, combine train data for full history
                        if series_id_col:
                            df_train_series = df_train[df_train[series_id_col] == selected_series].sort_values(date_col).reset_index(drop=True)
                            df_score_series = df_score[df_score[series_id_col] == selected_series].sort_values(date_col).reset_index(drop=True)
                            
                            if len(df_score_series) == 0:
                                st.error(f"Series '{selected_series}' not found in score dataset.")
                                st.stop()
                            
                            n_train_points = len(df_train_series)
                            n_score_points = len(df_score_series)
                            
                            st.info(f"Training on series '{selected_series}' from _train ({n_train_points} points), predicting on _score ({n_score_points} points)")
                            
                            # Prepare train TimeSeriesDataFrame
                            df_train_series['_series_id'] = selected_series
                            train_tsdf = pandas_to_time_series_dataframe(
                                df_train_series,
                                item_id_col='_series_id',
                                timestamp_col=date_col,
                                target_col=target_col
                            )
                            
                            # Generate test X for the score period length
                            test_X = generate_test_X(train_tsdf, n_score_points)
                            
                        else:
                            df_train_sorted = df_train.sort_values(date_col).reset_index(drop=True)
                            df_score_sorted = df_score.sort_values(date_col).reset_index(drop=True)
                            
                            n_train_points = len(df_train_sorted)
                            n_score_points = len(df_score_sorted)
                            
                            # Prepare train TimeSeriesDataFrame
                            df_train_sorted['_series_id'] = 'single_series'
                            train_tsdf = pandas_to_time_series_dataframe(
                                df_train_sorted,
                                item_id_col='_series_id',
                                timestamp_col=date_col,
                                target_col=target_col
                            )
                            
                            # Generate test X for the score period length
                            test_X = generate_test_X(train_tsdf, n_score_points)
                        
                        # Run forecasting with TabPFNTimeSeriesPredictor
                        results = run_forecasting_tabpfn_ts(
                            train_tsdf, test_X, None, n_score_points
                        )
                        
                        # Get dates for visualization
                        train_dates_viz = train_tsdf.index.get_level_values('timestamp')
                        train_values_viz = train_tsdf['target'].values
                        
                        # Get score dates from the score dataset
                        if series_id_col:
                            score_dates_subset = pd.to_datetime(df_score_series[date_col])
                        else:
                            score_dates_subset = pd.to_datetime(df_score_sorted[date_col])
                        
                        # Adjust predictions to match score dates
                        predictions_len = min(len(results['predictions']), len(score_dates_subset))
                        
                        if series_id_col:
                            predictions_df = pd.DataFrame({
                                "Series_ID": selected_series,
                                "Date": score_dates_subset[:predictions_len].dt.strftime('%Y-%m-%d'),
                                "Prediction": results['predictions'][:predictions_len]
                            })
                        else:
                            predictions_df = pd.DataFrame({
                                "Date": score_dates_subset[:predictions_len].dt.strftime('%Y-%m-%d'),
                                "Prediction": results['predictions'][:predictions_len]
                            })
                        
                        st.markdown('<div class="section-header">📊 Forecast Predictions on Score Data</div>', unsafe_allow_html=True)
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(12, 5))
                        fig.patch.set_facecolor('#f8f7ff')
                        ax.set_facecolor('#f8f7ff')
                        
                        # Plot training data (historical)
                        ax.plot(train_dates_viz, train_values_viz, color='#667eea', linewidth=1.5, label='Historical (Train)', alpha=0.7)
                        
                        # Plot predictions for score period
                        ax.plot(score_dates_subset[:predictions_len], results['predictions'][:predictions_len], color='#e056a0', linewidth=2, marker='s', markersize=6, linestyle='--', label='Forecast (Score)')
                        
                        # Add prediction intervals if available
                        if results.get('y_lower') is not None:
                            ax.fill_between(score_dates_subset[:predictions_len], results['y_lower'][:predictions_len], results['y_upper'][:predictions_len], alpha=0.15, color='#e056a0', label='80% Interval')
                        
                        # Add vertical line at train/score boundary
                        ax.axvline(x=train_dates_viz.max(), color='#00b894', linestyle=':', linewidth=2, alpha=0.7, label='Train/Score Split')
                        
                        ax.set_xlabel('Date', color='#2d2d44')
                        ax.set_ylabel(target_col, color='#2d2d44')
                        title = f'Forecast: {selected_series}' if series_id_col else 'Forecast Predictions'
                        ax.set_title(f'{title} (TabPFNTimeSeriesPredictor)', color='#1a1a2e', fontweight='bold')
                        ax.legend(facecolor='#ffffff', edgecolor='#e0e0e0', labelcolor='#2d2d44')
                        ax.tick_params(colors='#4a4a6a')
                        ax.grid(True, alpha=0.3, color='#667eea')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Training Samples", len(train_values_viz))
                        with col2:
                            st.metric("Score Samples", len(predictions_df))
                        
                        st.dataframe(predictions_df, use_container_width=True)
                        
                        # Log to MLflow
                        run_id, run_url, mlflow_err = log_to_mlflow(
                            run_name=f"app_{selected_base_table}_forecast_score",
                            params={
                                "dataset": selected_base_table,
                                "model_type": "TabPFNTimeSeriesPredictor",
                                "tabpfn_mode": "CLIENT",
                                "task": "time_series_forecasting",
                                "target_col": target_col,
                                "date_col": date_col,
                                "series_id": selected_series if series_id_col else "single_series",
                                "forecast_horizon": forecast_horizon,
                                "train_samples": len(train_values_viz),
                                "score_samples": len(predictions_df),
                                "features_used": str(results.get('features_used', [])),
                                "random_state": random_state,
                            },
                            metrics={},
                            task_type="forecast",
                            operation_mode="score",
                            mlflow_config=mlflow_config
                        )
                        if run_id:
                            if run_url:
                                st.write(f"📋 MLflow Run: {run_id}")
                                st.link_button("View in MLflow", run_url)
                            else:
                                st.caption(f"📋 MLflow Run ID: `{run_id}`")
                        elif mlflow_err:
                            st.caption(f"⚠️ MLflow logging failed: {mlflow_err}")
                        
                        # Download button
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"{selected_base_table}_forecast_predictions.csv",
                            mime="text/csv"
                        )
                    
                    st.success("✅ Forecasting complete!")
                
                # ============================================================
                # CLASSIFICATION / REGRESSION
                # ============================================================
                else:
                    if operation_mode == "Evaluate":
                        # Prepare features from training data
                        X, y, encoded_cols = prepare_features(df_train, feature_cols, target_col)
                        
                        # Sample if needed
                        if max_samples and len(X) > max_samples:
                            np.random.seed(random_state)
                            idx = np.random.choice(len(X), max_samples, replace=False)
                            X, y = X[idx], y[idx]
                        
                        # Split for evaluation
                        from sklearn.model_selection import train_test_split
                        stratify = y if prediction_mode == "Classification" else None
                        X_train_model, X_test_model, y_train_model, y_test_model = train_test_split(
                            X, y, test_size=test_size/100, random_state=random_state, stratify=stratify
                        )
                        
                        if prediction_mode == "Classification":
                            results = run_classification(X_train_model, y_train_model, X_test_model, y_test_model)
                            
                            st.markdown('<div class="section-header">📊 Classification Evaluation Results</div>', unsafe_allow_html=True)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{results['accuracy']:.4f}")
                            with col2:
                                if results.get('roc_auc'):
                                    st.metric("ROC AUC", f"{results['roc_auc']:.4f}")
                            with col3:
                                st.metric("Test Samples", len(y_test_model))
                            
                            # Visualization: Confusion Matrix
                            y_true = results["y_test"]
                            y_pred = results["predictions"]
                            cm = confusion_matrix(y_true, y_pred)
                            classes = np.unique(np.concatenate([y_true, y_pred]))
                            
                            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                            fig.patch.set_facecolor('#f8f7ff')
                            
                            # Confusion Matrix Heatmap
                            ax1 = axes[0]
                            ax1.set_facecolor('#f8f7ff')
                            im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
                            ax1.figure.colorbar(im, ax=ax1, shrink=0.8)
                            ax1.set(xticks=np.arange(cm.shape[1]),
                                    yticks=np.arange(cm.shape[0]),
                                    xticklabels=classes, yticklabels=classes,
                                    ylabel='Actual', xlabel='Predicted')
                            ax1.set_title('Confusion Matrix', color='#1a1a2e', fontweight='bold')
                            ax1.tick_params(colors='#4a4a6a')
                            
                            # Add text annotations
                            thresh = cm.max() / 2.
                            for i in range(cm.shape[0]):
                                for j in range(cm.shape[1]):
                                    ax1.text(j, i, format(cm[i, j], 'd'),
                                            ha="center", va="center",
                                            color="white" if cm[i, j] > thresh else "black",
                                            fontsize=12, fontweight='bold')
                            
                            # Class Distribution Bar Chart
                            ax2 = axes[1]
                            ax2.set_facecolor('#f8f7ff')
                            
                            # Count actual vs predicted for each class
                            unique_classes = sorted(classes)
                            actual_counts = [np.sum(y_true == c) for c in unique_classes]
                            predicted_counts = [np.sum(y_pred == c) for c in unique_classes]
                            
                            x = np.arange(len(unique_classes))
                            width = 0.35
                            
                            bars1 = ax2.bar(x - width/2, actual_counts, width, label='Actual', color='#667eea', alpha=0.8)
                            bars2 = ax2.bar(x + width/2, predicted_counts, width, label='Predicted', color='#e056a0', alpha=0.8)
                            
                            ax2.set_xlabel('Class', color='#2d2d44')
                            ax2.set_ylabel('Count', color='#2d2d44')
                            ax2.set_title('Class Distribution: Actual vs Predicted', color='#1a1a2e', fontweight='bold')
                            ax2.set_xticks(x)
                            ax2.set_xticklabels([str(c) for c in unique_classes])
                            ax2.legend(facecolor='#ffffff', edgecolor='#e0e0e0', labelcolor='#2d2d44')
                            ax2.tick_params(colors='#4a4a6a')
                            ax2.grid(True, alpha=0.3, color='#667eea', axis='y')
                            
                            # Add value labels on bars
                            for bar in bars1:
                                height = bar.get_height()
                                ax2.annotate(f'{int(height)}',
                                            xy=(bar.get_x() + bar.get_width() / 2, height),
                                            xytext=(0, 3), textcoords="offset points",
                                            ha='center', va='bottom', fontsize=10, color='#2d2d44')
                            for bar in bars2:
                                height = bar.get_height()
                                ax2.annotate(f'{int(height)}',
                                            xy=(bar.get_x() + bar.get_width() / 2, height),
                                            xytext=(0, 3), textcoords="offset points",
                                            ha='center', va='bottom', fontsize=10, color='#2d2d44')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Results dataframe
                            results_df = pd.DataFrame({
                                "Actual": results["y_test"],
                                "Predicted": results["predictions"],
                                "Correct": results["y_test"] == results["predictions"]
                            })
                            
                            # Add probabilities
                            n_classes = results["probabilities"].shape[1]
                            for i in range(n_classes):
                                results_df[f"Prob_Class_{i}"] = results["probabilities"][:, i].round(4)
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Log to MLflow
                            run_id, run_url, mlflow_err = log_to_mlflow(
                                run_name=f"app_{selected_base_table}_classification_eval",
                                params={
                                    "dataset": selected_base_table,
                                    "model_type": "TabPFNClassifier",
                                    "target_col": target_col,
                                    "n_features": len(feature_cols),
                                    "train_samples": len(X_train_model),
                                    "test_samples": len(X_test_model),
                                    "test_size_pct": test_size,
                                    "random_state": random_state,
                                    "n_classes": n_classes,
                                },
                                metrics={
                                    "accuracy": results['accuracy'],
                                    "roc_auc": results.get('roc_auc'),
                                },
                                task_type="classification",
                                operation_mode="evaluate",
                                mlflow_config=mlflow_config
                            )
                            if run_id:
                                if run_url:
                                    st.write(f"📋 MLflow Run: {run_id}")
                                    st.link_button("View in MLflow", run_url)
                                else:
                                    st.caption(f"📋 MLflow Run ID: `{run_id}`")
                            elif mlflow_err:
                                st.caption(f"⚠️ MLflow logging failed: {mlflow_err}")
                        
                        else:  # Regression
                            results = run_regression(X_train_model, y_train_model, X_test_model, y_test_model)
                            
                            st.markdown('<div class="section-header">📊 Regression Evaluation Results</div>', unsafe_allow_html=True)
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("RMSE", f"{results['rmse']:.4f}")
                            with col2:
                                st.metric("MAE", f"{results['mae']:.4f}")
                            with col3:
                                st.metric("R²", f"{results['r2']:.4f}")
                            with col4:
                                st.metric("Test Samples", len(y_test_model))
                            
                            # Visualization
                            y_true = results["y_test"]
                            y_pred = results["predictions"]
                            residuals = y_true - y_pred
                            
                            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                            fig.patch.set_facecolor('#f8f7ff')
                            
                            # Actual vs Predicted Scatter Plot
                            ax1 = axes[0]
                            ax1.set_facecolor('#f8f7ff')
                            ax1.scatter(y_true, y_pred, alpha=0.6, color='#667eea', edgecolors='white', linewidth=0.5, s=60)
                            
                            # Perfect prediction line
                            min_val = min(y_true.min(), y_pred.min())
                            max_val = max(y_true.max(), y_pred.max())
                            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction', color='#e056a0')
                            
                            ax1.set_xlabel('Actual', color='#2d2d44')
                            ax1.set_ylabel('Predicted', color='#2d2d44')
                            ax1.set_title('Actual vs Predicted', color='#1a1a2e', fontweight='bold')
                            ax1.legend(facecolor='#ffffff', edgecolor='#e0e0e0', labelcolor='#2d2d44')
                            ax1.tick_params(colors='#4a4a6a')
                            ax1.grid(True, alpha=0.3, color='#667eea')
                            
                            # Residuals Distribution
                            ax2 = axes[1]
                            ax2.set_facecolor('#f8f7ff')
                            ax2.hist(residuals, bins=30, color='#667eea', alpha=0.7, edgecolor='white')
                            ax2.axvline(x=0, color='#e056a0', linestyle='--', linewidth=2, label='Zero Error')
                            ax2.axvline(x=residuals.mean(), color='#00b894', linestyle='-', linewidth=2, label=f'Mean: {residuals.mean():.2f}')
                            
                            ax2.set_xlabel('Residual (Actual - Predicted)', color='#2d2d44')
                            ax2.set_ylabel('Frequency', color='#2d2d44')
                            ax2.set_title('Residuals Distribution', color='#1a1a2e', fontweight='bold')
                            ax2.legend(facecolor='#ffffff', edgecolor='#e0e0e0', labelcolor='#2d2d44')
                            ax2.tick_params(colors='#4a4a6a')
                            ax2.grid(True, alpha=0.3, color='#667eea', axis='y')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Results dataframe
                            results_df = pd.DataFrame({
                                "Actual": y_true,
                                "Predicted": y_pred,
                                "Residual": residuals,
                                "Abs Error": np.abs(residuals)
                            })
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Log to MLflow
                            run_id, run_url, mlflow_err = log_to_mlflow(
                                run_name=f"app_{selected_base_table}_regression_eval",
                                params={
                                    "dataset": selected_base_table,
                                    "model_type": "TabPFNRegressor",
                                    "target_col": target_col,
                                    "n_features": len(feature_cols),
                                    "train_samples": len(X_train_model),
                                    "test_samples": len(X_test_model),
                                    "test_size_pct": test_size,
                                    "random_state": random_state,
                                },
                                metrics={
                                    "rmse": results['rmse'],
                                    "mae": results['mae'],
                                    "r2": results['r2'],
                                },
                                task_type="regression",
                                operation_mode="evaluate",
                                mlflow_config=mlflow_config
                            )
                            if run_id:
                                if run_url:
                                    st.write(f"📋 MLflow Run: {run_id}")
                                    st.link_button("View in MLflow", run_url)
                                else:
                                    st.caption(f"📋 MLflow Run ID: `{run_id}`")
                            elif mlflow_err:
                                st.caption(f"⚠️ MLflow logging failed: {mlflow_err}")
                        
                        st.success("✅ Evaluation complete!")
                    
                    else:  # Score mode
                        # Load score data
                        with st.spinner(f"Loading {score_table}..."):
                            df_score = load_table(conn, f"{CATALOG}.{SCHEMA}.{score_table}")
                        
                        # Prepare training features
                        X_train_full, y_train_full, encoded_cols = prepare_features(df_train, feature_cols, target_col)
                        
                        # Sample if needed
                        if max_samples and len(X_train_full) > max_samples:
                            np.random.seed(random_state)
                            idx = np.random.choice(len(X_train_full), max_samples, replace=False)
                            X_train_full, y_train_full = X_train_full[idx], y_train_full[idx]
                        
                        # Prepare scoring features (same encoding)
                        X_score, _, _ = prepare_features(df_score, feature_cols, target_col)
                        
                        if prediction_mode == "Classification":
                            results = run_classification(X_train_full, y_train_full, X_score)
                            
                            st.markdown('<div class="section-header">📊 Classification Predictions</div>', unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Training Samples", len(X_train_full))
                            with col2:
                                st.metric("Scored Samples", len(X_score))
                            
                            # Results dataframe - features and predictions only (exclude target column)
                            display_cols = [col for col in df_score.columns if col != target_col]
                            results_df = df_score[display_cols].copy()
                            results_df["Prediction"] = results["predictions"]
                            
                            # Add probabilities
                            n_classes = results["probabilities"].shape[1]
                            for i in range(n_classes):
                                results_df[f"Prob_Class_{i}"] = results["probabilities"][:, i].round(4)
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Log to MLflow
                            run_id, run_url, mlflow_err = log_to_mlflow(
                                run_name=f"app_{selected_base_table}_classification_score",
                                params={
                                    "dataset": selected_base_table,
                                    "model_type": "TabPFNClassifier",
                                    "target_col": target_col,
                                    "n_features": len(feature_cols),
                                    "train_samples": len(X_train_full),
                                    "score_samples": len(X_score),
                                    "random_state": random_state,
                                    "n_classes": n_classes,
                                },
                                metrics={},
                                task_type="classification",
                                operation_mode="score",
                                mlflow_config=mlflow_config
                            )
                            if run_id:
                                if run_url:
                                    st.write(f"📋 MLflow Run: {run_id}")
                                    st.link_button("View in MLflow", run_url)
                                else:
                                    st.caption(f"📋 MLflow Run ID: `{run_id}`")
                            elif mlflow_err:
                                st.caption(f"⚠️ MLflow logging failed: {mlflow_err}")
                        
                        else:  # Regression
                            results = run_regression(X_train_full, y_train_full, X_score)
                            
                            st.markdown('<div class="section-header">📊 Regression Predictions</div>', unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Training Samples", len(X_train_full))
                            with col2:
                                st.metric("Scored Samples", len(X_score))
                            
                            # Results dataframe - features and predictions only (exclude target column)
                            display_cols = [col for col in df_score.columns if col != target_col]
                            results_df = df_score[display_cols].copy()
                            results_df["Prediction"] = results["predictions"]
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Log to MLflow
                            run_id, run_url, mlflow_err = log_to_mlflow(
                                run_name=f"app_{selected_base_table}_regression_score",
                                params={
                                    "dataset": selected_base_table,
                                    "model_type": "TabPFNRegressor",
                                    "target_col": target_col,
                                    "n_features": len(feature_cols),
                                    "train_samples": len(X_train_full),
                                    "score_samples": len(X_score),
                                    "random_state": random_state,
                                },
                                metrics={},
                                task_type="regression",
                                operation_mode="score",
                                mlflow_config=mlflow_config
                            )
                            if run_id:
                                if run_url:
                                    st.write(f"📋 MLflow Run: {run_id}")
                                    st.link_button("View in MLflow", run_url)
                                else:
                                    st.caption(f"📋 MLflow Run ID: `{run_id}`")
                            elif mlflow_err:
                                st.caption(f"⚠️ MLflow logging failed: {mlflow_err}")
                        
                        st.success("✅ Scoring complete!")
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"{selected_base_table}_predictions.csv",
                            mime="text/csv"
                        )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
