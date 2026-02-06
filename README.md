# TabPFN on Databricks

[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Serverless](https://img.shields.io/badge/Serverless-Compute-00C851?style=for-the-badge)](https://docs.databricks.com/en/compute/serverless.html)

A comprehensive solution accelerator demonstrating how to use **TabPFN** (Tabular Prior-Data Fitted Network) on Databricks.

## Overview

### The Challenge: Enterprise-Scale Predictive Analytics

Across industries, enterprises rely on large-scale tabular prediction to drive critical business decisions—from forecasting demand to assessing risk to optimizing operations. These predictive workloads often span interconnected business processes, each requiring its own set of models, features, and continuous maintenance.

Consider a global retail and consumer packaged goods (CPG) company, where demand and supply planning spans many interconnected business processes:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Demand Planning │───▶│ Supply Planning │───▶│ Production      │───▶│ Distribution    │
│                 │    │                 │    │ Planning        │    │ Planning        │
│ • Forecasting   │    │ • Supplier Risk │    │ • Yield Pred.   │    │ • Transport LT  │
│ • Price Elast.  │    │ • Lead Time     │    │ • Scrap Detect. │    │ • OTIF Risk     │
│ • Promo Lift    │    │ • Material      │    │ • Capacity      │    │                 │
│                 │    │   Shortage      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

This value chain transforms customer demand signals into fulfilled orders. Demand planning generates forecasts that drive supply planning decisions; supply planning determines what to source and when, feeding production schedules; and production output flows into distribution planning for delivery to stores and customers. At each stage, predictive models provide critical inputs—and errors propagate downstream.

For each of these processes, there is typically one or more teams of data scientists responsible for:
- **Sourcing and cleaning data** from disparate systems
- **Analyzing patterns** and building features
- **Training models** and tuning hyperparameters
- **Deploying and monitoring** models in production
- **Continuously evaluating** performance and retraining

At enterprise scale, this creates **significant operational overhead**. Organizations often maintain large model portfolios—potentially **thousands or even millions of models** across products, SKUs, regions, customers, and planning levels. Maintaining this at high quality requires substantial human effort, and hiring and retaining large teams of experienced data scientists is **costly**.

### The Solution: Foundation Models for Tabular Data

**Imagine a pretrained model that:**
- Works **out of the box** on tabular data
- Requires **minimal preprocessing**
- Eliminates the need for **model training, hyperparameter tuning, and complex experimentation**
- Delivers performance **comparable to, or better than**, many carefully tuned traditional models

This is exactly what [**TabPFN**](https://docs.priorlabs.ai/overview/) enables.

TabPFN (Tabular Prior-Data Fitted Network) is a **foundation model for tabular prediction** developed by Prior Labs. It has been pretrained on hundreds of millions of synthetic datasets to learn general patterns in tabular data, allowing it to make exceptionally accurate predictions on new datasets **without any training**.

**Key benefits:**
- **Reduced operational complexity**: One model for many use cases
- **Zero training time**: Predictions in seconds
- **No hyperparameter tuning**: Works out of the box
- **Strong default performance**: Competitive with tuned XGBoost, Random Forest, and other models
- **Built-in uncertainty quantification**: Get prediction intervals, not just point estimates

## Use Cases (Retail/CPG)

This solution accelerator demonstrates TabPFN across the retail/CPG planning value chain:

| Use Case | ML Task | Planning Process | Business Value |
|----------|---------|------------------|----------------|
| **Demand Forecasting** | Time Series | Demand Planning | Inventory & capacity planning |
| **Price Elasticity** | Regression | Demand Planning | Optimize pricing strategies |
| **Promotion Lift** | Regression | Demand Planning | Optimize trade promotion ROI |
| **Supplier Delay Risk** | Binary Classification | Supply Planning | Proactive risk mitigation |
| **Supplier Lead Time** | Regression | Supply Planning | Improve planning accuracy |
| **Material Shortage** | Multi-class Classification | Supply Planning | Prioritize procurement actions |
| **Labor Shortage** | Multi-class Classification | Production Planning | Workforce planning |
| **Yield Prediction** | Regression | Production Planning | Capacity planning |
| **Scrap Anomaly Detection** | Anomaly Detection | Production Planning | Early quality issue detection |
| **Transportation Lead Time** | Regression | Distribution Planning | Delivery planning |
| **OTIF Risk** | Multi-class Classification | Distribution Planning | Proactive customer service |

## Features

| Feature | Description |
|---------|-------------|
| **Classification** | Binary and multi-class classification with probability estimates |
| **Regression** | Continuous value prediction with uncertainty quantification |
| **Outlier Detection** | Anomaly scoring using semi-supervised learning |
| **Time Series Forecasting** | Lag-based forecasting with TabPFN Regressor |
| **Unity Catalog Integration** | Read/write data from Delta tables |
| **Databricks App** | Interactive Streamlit UI for predictions |
| **Realistic Datasets** | Synthetic retail/CPG data with business-relevant features |

## Beyond Retail/CPG

This paradigm shift is **not limited to retail and CPG**. The same opportunity applies broadly across industries with large-scale tabular prediction needs:

| Industry | Example Use Cases |
|----------|-------------------|
| **Financial Services (FSI)** | Credit scoring, fraud detection, churn prediction, risk modeling |
| **Manufacturing (MFG)** | Predictive maintenance, quality prediction, yield optimization |
| **Health & Life Sciences (HLS)** | Patient risk stratification, clinical trial optimization, drug discovery |
| **Energy & Utilities** | Demand forecasting, outage prediction, asset management |
| **Telecommunications** | Network optimization, customer lifetime value, service quality |

## Project Structure

```
tabpfn-databricks/
├── apps/                           # Databricks App (Streamlit)
│   ├── app.py                      # Main Streamlit application
│   ├── app.yaml                    # App deployment configuration
│   └── requirements.txt            # App dependencies
├── notebooks/                      # Jupyter notebooks
│   ├── 00_data_preparation.ipynb   # Generate retail/CPG datasets
│   ├── 01_classification.ipynb     # Supplier delay & shortage prediction
│   ├── 02_regression.ipynb         # Price elasticity & promotion lift
│   ├── 03_outlier_detection.ipynb  # Production scrap anomaly detection
│   └── 04_time_series_forecasting.ipynb  # Demand forecasting
├── scripts/                        # Utility scripts
│   ├── util.py                     # Data generation functions
│   └── cleanup.sh                  # Resource cleanup
├── .github/workflows/              # CI/CD pipelines
│   └── databricks-ci.yml           # Databricks Asset Bundle CI
├── dashboards/                     # Databricks dashboards (placeholder)
├── databricks.yml                  # Databricks Asset Bundle configuration
├── requirements.txt                # Project dependencies
├── CONTRIBUTING.md                 # Contribution guidelines
├── LICENSE.md                      # Databricks license
└── SECURITY.md                     # Security policy
```

## Prerequisites

1. **Databricks Workspace** with Unity Catalog enabled
2. **TabPFN API Token** from [Prior Labs](https://docs.priorlabs.ai/)
3. **Databricks CLI** (optional, for local development)

## Getting Started

### Option 1: Clone via Databricks UI (Recommended)

1. **Clone the repository** into your Databricks Workspace:
   - Navigate to **Repos** in the Databricks UI
   - Click **Add Repo** and enter the repository URL

2. **Open the Asset Bundle Editor**:
   - Click on the workspace folder containing the cloned repo
   - Open the Asset Bundle Editor from the UI

3. **Deploy the bundle**:
   - Click **Deploy** to deploy all resources

4. **Run the notebooks**:
   - Navigate to the Deployments tab (🚀 icon)
   - Click **Run** to execute the workflow

### Option 2: Deploy via Databricks CLI

1. **Install the Databricks CLI**:
   ```bash
   pip install databricks-cli
   ```

2. **Configure authentication**:
   ```bash
   databricks configure --token
   ```

3. **Clone and deploy**:
   ```bash
   git clone <repository-url>
   cd tabpfn-databricks
   databricks bundle deploy
   ```

### Setting Up TabPFN Authentication

Store your TabPFN token as a Databricks Secret:

```bash
# Create a secret scope
databricks secrets create-scope tabpfn-client

# Store your token
databricks secrets put-secret tabpfn-client token
```

To retrieve your token on another machine:
```python
import tabpfn_client
token = tabpfn_client.get_access_token()
print(token)
```

## Notebooks Overview

### 00_data_preparation.ipynb
Generates realistic retail/CPG planning datasets and stores them as Delta tables in Unity Catalog.

**Datasets:**

| Table | Task | Planning Process | Samples |
|-------|------|------------------|---------|
| `demand_forecast` | Time Series | Demand Planning | 1,800 |
| `price_elasticity` | Regression | Demand Planning | 3,000 |
| `promotion_lift` | Regression | Demand Planning | 2,500 |
| `supplier_delay_risk` | Binary Classification | Supply Planning | 2,000 |
| `supplier_lead_time` | Regression | Supply Planning | 2,000 |
| `material_shortage` | Multi-class Classification | Supply Planning | 1,500 |
| `labor_shortage` | Multi-class Classification | Production Planning | 1,500 |
| `yield_prediction` | Regression | Production Planning | 2,000 |
| `scrap_anomaly` | Anomaly Detection | Production Planning | 1,000 |
| `transportation_lead_time` | Regression | Distribution Planning | 2,000 |
| `otif_risk` | Multi-class Classification | Distribution Planning | 2,000 |

### 01_classification.ipynb
Demonstrates supply chain risk classification:
- **Supplier Delay Risk** (Supply Planning): Predict which deliveries will be delayed
- **Material Shortage** (Supply Planning): Predict shortage risk levels (No Risk, At Risk, Critical)
- **Labor Shortage** (Production Planning): Predict workforce availability issues
- **OTIF Risk** (Distribution Planning): Predict on-time-in-full delivery risk (Low, Medium, High Risk)
- Model comparison with Random Forest, Gradient Boosting, Logistic Regression
- Business impact quantification

### 02_regression.ipynb
Shows regression with uncertainty across the planning value chain:
- **Price Elasticity** (Demand Planning): Predict price sensitivity by product/market
- **Promotion Lift** (Demand Planning): Forecast incremental sales from promotions
- **Supplier Lead Time** (Supply Planning): Predict actual delivery times
- **Yield Prediction** (Production Planning): Predict production output yield
- **Transportation Lead Time** (Distribution Planning): Predict shipment transit times
- 80% prediction intervals using quantile regression
- Model comparison and business applications

### 03_outlier_detection.ipynb
Demonstrates production anomaly detection:
- Semi-supervised anomaly detection using TabPFN
- Comparison with Isolation Forest and Local Outlier Factor
- Business value: Early detection of equipment issues, quality problems
- Feature importance analysis for anomaly patterns

### 04_time_series_forecasting.ipynb
Shows demand forecasting with lag features:
- Feature engineering with lag and calendar variables
- Uncertainty quantification for safety stock planning
- Batch forecasting across multiple series
- Aggregate forecast reconciliation

## Databricks App

The project includes a Streamlit-based Databricks App for interactive supply chain analytics.

### Features
- Select planning use cases from dropdown
- Run TabPFN predictions on Unity Catalog data
- View predictions with business context
- Identify high-risk items for proactive action

### Configuration

The app is configured via `apps/app.yaml`. Before deploying, update the environment variables:

```yaml
env:
  - name: DATABRICKS_HTTP_PATH
    value: "/sql/1.0/warehouses/your_warehouse_id"
  - name: TABPFN_TOKEN
    value: "your_tabpfn_token"
```

| Environment Variable | Description | How to Find |
|---------------------|-------------|-------------|
| `DATABRICKS_HTTP_PATH` | SQL Warehouse HTTP path | SQL Warehouses → Select warehouse → Connection details |
| `TABPFN_TOKEN` | TabPFN API token | Run `tabpfn_client.get_access_token()` or visit [Prior Labs](https://docs.priorlabs.ai/) |

### Running Locally

```bash
cd apps
pip install -r requirements.txt

# Set environment variables
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/your_warehouse_id"
export TABPFN_TOKEN="your_tabpfn_token"

streamlit run app.py
```

## Data Generation

The `scripts/util.py` module provides functions to generate realistic retail/CPG datasets:

```python
from util import (
    # Classification
    generate_supplier_delay_risk_data,
    generate_material_shortage_data,
    generate_labor_shortage_data,
    generate_otif_risk_data,
    # Regression
    generate_price_elasticity_data,
    generate_promotion_lift_data,
    generate_supplier_lead_time_data,
    generate_transportation_lead_time_data,
    generate_yield_prediction_data,
    # Anomaly Detection
    generate_scrap_anomaly_data,
    # Time Series
    generate_aggregate_demand_forecast_data,
)

# Generate supplier delay risk data
df = generate_supplier_delay_risk_data(n_samples=2000, seed=42)
```

Each dataset includes realistic features such as:
- SKU, store, DC, region identifiers
- Calendar features (month, week, season)
- Pricing and promotion attributes
- Inventory and lead time metrics
- Supplier characteristics
- Production parameters

## Compute Requirements

All notebooks are optimized for **Serverless Compute** with **Base Environment V4**:

1. Click on the compute selector in the notebook toolbar
2. Select **Serverless**
3. Under Environment, choose **Base Environment V4**

Serverless compute provides fast startup times and automatic scaling, ideal for interactive notebook workflows.

## CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/databricks-ci.yml`) that:

1. Validates the Databricks Asset Bundle
2. Deploys resources to the workspace
3. Runs the demo workflow
4. Cleans up PR deployments

### Required Secrets

| Secret | Description |
|--------|-------------|
| `DEPLOY_NOTEBOOK_TOKEN` | Databricks personal access token |

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tabpfn-client` | >=0.1.0 | TabPFN API client |
| `databricks-sdk` | >=0.20.0 | Databricks SDK |
| `databricks-sql-connector` | >=3.0.0 | SQL connectivity |
| `scikit-learn` | >=1.3.0 | ML utilities |
| `pandas` | >=2.0.0 | Data manipulation |
| `numpy` | >=1.24.0 | Numerical computing |
| `streamlit` | >=1.30.0 | Web app framework |

## Resources

- [TabPFN Documentation](https://docs.priorlabs.ai/)
- [TabPFN Client GitHub](https://github.com/PriorLabs/tabpfn-client)
- [Prior Labs Website](https://priorlabs.ai/)
- [Databricks Asset Bundles](https://docs.databricks.com/en/dev-tools/bundles/index.html)
- [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and CLA information.

## Third-Party Package Licenses

| Package | License | Copyright |
|---------|---------|-----------|
| tabpfn-client | Apache 2.0 | Prior Labs GmbH |
| scikit-learn | BSD-3-Clause | scikit-learn developers |
| pandas | BSD-3-Clause | pandas development team |
| numpy | BSD-3-Clause | NumPy developers |
| streamlit | Apache 2.0 | Streamlit Inc. |
| databricks-sdk | Apache 2.0 | Databricks, Inc. |

## License

This project is provided subject to the [Databricks License](LICENSE.md).

---

© 2025 Databricks, Inc. All rights reserved.
