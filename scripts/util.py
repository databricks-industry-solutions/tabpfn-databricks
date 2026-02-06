"""
Retail/CPG Planning Data Generation Utilities

This module generates realistic synthetic datasets for retail/CPG planning
analytics demonstrations, covering the end-to-end value chain from demand
planning to supply planning.

Use Cases Covered:
- Classification: Supplier delay risk, material shortage prediction, labor shortage prediction, OTIF risk prediction
- Regression: Price elasticity, promotion lift, supplier lead time, transportation lead time, yield prediction
- Anomaly Detection: Scrap/defect detection, production anomalies
- Time Series: Demand forecasting by SKU/Store/DC
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional


# =============================================================================
# Reference Data Generation
# =============================================================================

def generate_product_hierarchy(n_categories: int = 5, n_subcategories_per: int = 4,
                                n_skus_per: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Generate a product hierarchy typical of retail/CPG.
    
    Returns:
        DataFrame with columns: sku_id, sku_name, subcategory, category, brand, unit_cost
    """
    np.random.seed(seed)
    
    categories = ['Beverages', 'Snacks', 'Dairy', 'Frozen Foods', 'Personal Care'][:n_categories]
    brands = ['PrimeBrand', 'ValueChoice', 'NaturalSelect', 'FreshDaily', 'HomeEssentials']
    
    records = []
    sku_counter = 1
    
    for cat in categories:
        for sub_idx in range(n_subcategories_per):
            subcategory = f"{cat}_{chr(65 + sub_idx)}"  # A, B, C, D
            for _ in range(n_skus_per):
                sku_id = f"SKU{sku_counter:06d}"
                brand = np.random.choice(brands)
                unit_cost = np.random.uniform(1.0, 50.0)
                
                records.append({
                    'sku_id': sku_id,
                    'sku_name': f"{brand} {subcategory} Item {sku_counter}",
                    'subcategory': subcategory,
                    'category': cat,
                    'brand': brand,
                    'unit_cost': round(unit_cost, 2)
                })
                sku_counter += 1
    
    return pd.DataFrame(records)


def generate_location_hierarchy(n_regions: int = 4, n_dcs_per: int = 2,
                                 n_stores_per: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Generate a location hierarchy (Region -> DC -> Store).
    
    Returns:
        DataFrame with columns: location_id, location_type, location_name, dc_id, region
    """
    np.random.seed(seed)
    
    regions = ['Northeast', 'Southeast', 'Midwest', 'West'][:n_regions]
    
    records = []
    loc_counter = 1
    
    for region in regions:
        # Distribution Centers
        for dc_idx in range(n_dcs_per):
            dc_id = f"DC{loc_counter:04d}"
            records.append({
                'location_id': dc_id,
                'location_type': 'DC',
                'location_name': f"{region} Distribution Center {dc_idx + 1}",
                'dc_id': dc_id,
                'region': region
            })
            
            # Stores under this DC
            for store_idx in range(n_stores_per):
                loc_counter += 1
                store_id = f"STR{loc_counter:04d}"
                records.append({
                    'location_id': store_id,
                    'location_type': 'Store',
                    'location_name': f"{region} Store {store_idx + 1}",
                    'dc_id': dc_id,
                    'region': region
                })
            loc_counter += 1
    
    return pd.DataFrame(records)


def generate_supplier_master(n_suppliers: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Generate supplier master data.
    
    Returns:
        DataFrame with columns: supplier_id, supplier_name, country, supplier_tier,
                               avg_lead_time_days, reliability_score
    """
    np.random.seed(seed)
    
    countries = ['USA', 'Mexico', 'China', 'Germany', 'India', 'Vietnam', 'Brazil']
    tiers = ['Strategic', 'Preferred', 'Approved', 'Conditional']
    
    records = []
    for i in range(n_suppliers):
        tier = np.random.choice(tiers, p=[0.1, 0.3, 0.4, 0.2])
        # Reliability correlates with tier
        base_reliability = {'Strategic': 0.9, 'Preferred': 0.8, 'Approved': 0.7, 'Conditional': 0.6}[tier]
        
        records.append({
            'supplier_id': f"SUP{i+1:04d}",
            'supplier_name': f"Supplier {i+1} Corp",
            'country': np.random.choice(countries),
            'supplier_tier': tier,
            'avg_lead_time_days': int(np.random.uniform(7, 45)),
            'reliability_score': round(base_reliability + np.random.uniform(-0.1, 0.1), 2)
        })
    
    return pd.DataFrame(records)


# =============================================================================
# Classification Datasets
# =============================================================================

def generate_supplier_delay_risk_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate dataset for predicting supplier delivery delays.
    
    Features include supplier attributes, order characteristics, and external factors.
    Target: is_delayed (0 = on-time, 1 = delayed)
    
    Use case: Supply Planning - Predict which supplier deliveries are at risk of delay
    """
    np.random.seed(seed)
    
    # Generate supplier-related features
    supplier_tiers = np.random.choice(['Strategic', 'Preferred', 'Approved', 'Conditional'], 
                                       n_samples, p=[0.1, 0.3, 0.4, 0.2])
    countries = np.random.choice(['USA', 'Mexico', 'China', 'Germany', 'India'], 
                                  n_samples, p=[0.3, 0.2, 0.25, 0.15, 0.1])
    
    # Supplier performance history
    historical_otd_rate = np.random.beta(8, 2, n_samples)  # On-time delivery rate
    supplier_capacity_util = np.random.uniform(0.5, 1.0, n_samples)
    
    # Order characteristics
    order_quantity = np.random.exponential(500, n_samples)
    order_value = order_quantity * np.random.uniform(5, 50, n_samples)
    lead_time_days = np.random.randint(7, 60, n_samples)
    is_expedited = np.random.binomial(1, 0.15, n_samples)
    
    # Calendar features
    order_month = np.random.randint(1, 13, n_samples)
    is_peak_season = ((order_month >= 10) | (order_month <= 2)).astype(int)
    days_to_need_date = np.random.randint(1, 90, n_samples)
    
    # External factors
    port_congestion_index = np.random.uniform(0, 1, n_samples)
    fuel_price_index = np.random.uniform(0.8, 1.5, n_samples)
    weather_risk_score = np.random.uniform(0, 1, n_samples)
    
    # Generate target based on realistic relationships
    delay_prob = (
        0.05  # base delay rate
        + 0.15 * (supplier_tiers == 'Conditional').astype(float)
        + 0.10 * (supplier_tiers == 'Approved').astype(float)
        + 0.08 * (countries == 'China').astype(float)
        + 0.05 * (countries == 'India').astype(float)
        - 0.15 * historical_otd_rate
        + 0.12 * supplier_capacity_util
        + 0.08 * is_peak_season
        + 0.15 * port_congestion_index
        + 0.05 * weather_risk_score
        - 0.05 * is_expedited  # expedited orders get priority
        + 0.10 * (lead_time_days > 30).astype(float)
        + np.random.normal(0, 0.05, n_samples)
    )
    delay_prob = np.clip(delay_prob, 0.01, 0.95)
    is_delayed = np.random.binomial(1, delay_prob)
    
    df = pd.DataFrame({
        'supplier_tier': supplier_tiers,
        'supplier_country': countries,
        'historical_otd_rate': np.round(historical_otd_rate, 3),
        'supplier_capacity_utilization': np.round(supplier_capacity_util, 3),
        'order_quantity': np.round(order_quantity, 0).astype(int),
        'order_value_usd': np.round(order_value, 2),
        'contracted_lead_time_days': lead_time_days,
        'is_expedited_order': is_expedited,
        'order_month': order_month,
        'is_peak_season': is_peak_season,
        'days_to_need_date': days_to_need_date,
        'port_congestion_index': np.round(port_congestion_index, 3),
        'fuel_price_index': np.round(fuel_price_index, 3),
        'weather_risk_score': np.round(weather_risk_score, 3),
        'is_delayed': is_delayed
    })
    
    return df


def generate_labor_shortage_data(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generate dataset for predicting labor shortage risk at facilities.
    
    Features include workforce metrics, demand signals, and external factors.
    Target: shortage_risk (0 = Adequate, 1 = At Risk, 2 = Critical)
    
    Use case: Production Planning - Predict labor availability issues
    """
    np.random.seed(seed)
    
    # Facility characteristics
    facility_types = np.random.choice(['Warehouse', 'Manufacturing', 'Distribution Center'],
                                       n_samples, p=[0.35, 0.40, 0.25])
    facility_size = np.random.choice(['Small', 'Medium', 'Large'], n_samples, p=[0.3, 0.45, 0.25])
    regions = np.random.choice(['Northeast', 'Southeast', 'Midwest', 'West'],
                                n_samples, p=[0.25, 0.25, 0.25, 0.25])
    
    # Current workforce metrics
    current_headcount = np.random.randint(50, 500, n_samples)
    target_headcount = current_headcount * np.random.uniform(0.9, 1.2, n_samples)
    headcount_ratio = current_headcount / target_headcount
    turnover_rate_monthly = np.random.beta(2, 10, n_samples)  # typically 2-15%
    avg_tenure_months = np.random.exponential(18, n_samples)
    open_positions = np.random.poisson(8, n_samples)
    
    # Demand and workload
    forecasted_volume_change = np.random.uniform(-0.2, 0.4, n_samples)
    overtime_hours_last_month = np.random.exponential(200, n_samples)
    absenteeism_rate = np.random.beta(2, 20, n_samples)
    
    # Labor market factors
    local_unemployment_rate = np.random.uniform(0.03, 0.12, n_samples)
    competitor_wage_ratio = np.random.uniform(0.85, 1.15, n_samples)  # our wage / market
    job_posting_response_rate = np.random.beta(3, 7, n_samples)
    
    # Seasonality
    month = np.random.randint(1, 13, n_samples)
    is_peak_hiring_season = ((month >= 9) & (month <= 11)).astype(int)  # pre-holiday
    
    # Training and pipeline
    training_pipeline_count = np.random.poisson(5, n_samples)
    avg_time_to_fill_days = np.random.exponential(30, n_samples)
    
    # Generate target based on realistic relationships
    shortage_score = (
        0.4  # base
        - 0.5 * (headcount_ratio - 0.9)  # understaffed = higher risk
        + 0.4 * turnover_rate_monthly * 10
        - 0.2 * (avg_tenure_months > 24).astype(float)
        + 0.02 * open_positions
        + 0.3 * np.clip(forecasted_volume_change, 0, 1)
        + 0.15 * (overtime_hours_last_month > 300).astype(float)
        + 0.2 * absenteeism_rate * 10
        - 0.3 * local_unemployment_rate * 5  # higher unemployment = easier hiring
        - 0.2 * (competitor_wage_ratio > 1.0).astype(float)  # paying above market
        - 0.15 * job_posting_response_rate
        + 0.1 * is_peak_hiring_season
        + 0.01 * avg_time_to_fill_days / 30
        - 0.02 * training_pipeline_count
        + np.random.normal(0, 0.1, n_samples)
    )
    
    # Convert to multi-class
    shortage_risk = np.where(shortage_score < 0.35, 0,  # Adequate
                            np.where(shortage_score < 0.6, 1, 2))  # At Risk / Critical
    
    df = pd.DataFrame({
        'facility_type': facility_types,
        'facility_size': facility_size,
        'region': regions,
        'current_headcount': current_headcount,
        'target_headcount': np.round(target_headcount, 0).astype(int),
        'headcount_ratio': np.round(headcount_ratio, 3),
        'turnover_rate_monthly': np.round(turnover_rate_monthly, 3),
        'avg_tenure_months': np.round(avg_tenure_months, 1),
        'open_positions': open_positions,
        'forecasted_volume_change_pct': np.round(forecasted_volume_change, 3),
        'overtime_hours_last_month': np.round(overtime_hours_last_month, 0).astype(int),
        'absenteeism_rate': np.round(absenteeism_rate, 3),
        'local_unemployment_rate': np.round(local_unemployment_rate, 3),
        'competitor_wage_ratio': np.round(competitor_wage_ratio, 3),
        'job_posting_response_rate': np.round(job_posting_response_rate, 3),
        'month': month,
        'is_peak_hiring_season': is_peak_hiring_season,
        'training_pipeline_count': training_pipeline_count,
        'avg_time_to_fill_days': np.round(avg_time_to_fill_days, 0).astype(int),
        'labor_shortage_risk': shortage_risk
    })
    
    return df


def generate_material_shortage_data(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generate dataset for predicting material shortages.
    
    Features include inventory levels, demand signals, and supply factors.
    Target: shortage_risk (0 = No risk, 1 = At risk, 2 = Critical)
    
    Use case: Material Planning - Predict which materials are at risk of shortage
    """
    np.random.seed(seed)
    
    # Material characteristics
    material_types = np.random.choice(['Raw Material', 'Packaging', 'Component', 'Finished Good'],
                                       n_samples, p=[0.35, 0.25, 0.25, 0.15])
    criticality = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.2, 0.3, 0.5])
    
    # Inventory metrics
    current_stock_days = np.random.exponential(15, n_samples)
    safety_stock_days = np.random.uniform(5, 20, n_samples)
    stock_coverage_ratio = current_stock_days / (safety_stock_days + 0.1)
    
    # Demand signals
    forecast_demand_units = np.random.exponential(1000, n_samples)
    demand_variability_cv = np.random.uniform(0.1, 0.8, n_samples)  # coefficient of variation
    demand_trend = np.random.uniform(-0.2, 0.3, n_samples)  # % change
    
    # Supply factors
    n_active_suppliers = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    supplier_reliability_avg = np.random.beta(7, 2, n_samples)
    inbound_pipeline_qty = np.random.exponential(500, n_samples)
    
    # Lead time factors
    avg_lead_time_days = np.random.randint(7, 90, n_samples)
    lead_time_variability = np.random.uniform(0.1, 0.5, n_samples)
    
    # External factors
    commodity_price_trend = np.random.uniform(-0.1, 0.2, n_samples)
    geopolitical_risk_score = np.random.uniform(0, 1, n_samples)
    
    # Generate target based on realistic relationships
    shortage_score = (
        0.5  # base
        - 0.4 * np.clip(stock_coverage_ratio, 0, 2) / 2
        + 0.3 * demand_variability_cv
        + 0.2 * np.clip(demand_trend, 0, 1)
        - 0.15 * (n_active_suppliers > 1).astype(float)
        - 0.2 * supplier_reliability_avg
        + 0.15 * (avg_lead_time_days > 30).astype(float)
        + 0.1 * lead_time_variability
        + 0.15 * geopolitical_risk_score
        + 0.2 * (criticality == 'A').astype(float)
        + np.random.normal(0, 0.1, n_samples)
    )
    
    # Convert to multi-class
    shortage_risk = np.where(shortage_score < 0.3, 0,  # No risk
                            np.where(shortage_score < 0.6, 1, 2))  # At risk / Critical
    
    df = pd.DataFrame({
        'material_type': material_types,
        'criticality_class': criticality,
        'current_stock_days': np.round(current_stock_days, 1),
        'safety_stock_days': np.round(safety_stock_days, 1),
        'stock_coverage_ratio': np.round(stock_coverage_ratio, 2),
        'forecast_demand_units': np.round(forecast_demand_units, 0).astype(int),
        'demand_variability_cv': np.round(demand_variability_cv, 3),
        'demand_trend_pct': np.round(demand_trend, 3),
        'num_active_suppliers': n_active_suppliers,
        'avg_supplier_reliability': np.round(supplier_reliability_avg, 3),
        'inbound_pipeline_qty': np.round(inbound_pipeline_qty, 0).astype(int),
        'avg_lead_time_days': avg_lead_time_days,
        'lead_time_variability': np.round(lead_time_variability, 3),
        'commodity_price_trend': np.round(commodity_price_trend, 3),
        'geopolitical_risk_score': np.round(geopolitical_risk_score, 3),
        'shortage_risk': shortage_risk
    })
    
    return df


def generate_otif_risk_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate dataset for predicting On-Time-In-Full (OTIF) delivery risk.
    
    Features include shipment characteristics, customer factors, and logistics conditions.
    Target: otif_risk (0 = Low Risk/Likely OTIF, 1 = Medium Risk, 2 = High Risk/Likely Failure)
    
    Use case: Distribution Planning - Predict which orders are at risk of not being delivered 
    on-time and in-full to enable proactive intervention
    """
    np.random.seed(seed)
    
    # Order characteristics
    order_types = np.random.choice(['Standard', 'Express', 'Same-Day', 'Scheduled'],
                                    n_samples, p=[0.5, 0.25, 0.1, 0.15])
    order_size = np.random.choice(['Small', 'Medium', 'Large', 'Bulk'],
                                   n_samples, p=[0.3, 0.4, 0.2, 0.1])
    n_line_items = np.random.poisson(5, n_samples) + 1
    order_value_usd = np.random.exponential(500, n_samples) + 50
    
    # Customer characteristics
    customer_tiers = np.random.choice(['Platinum', 'Gold', 'Silver', 'Bronze'],
                                       n_samples, p=[0.1, 0.25, 0.35, 0.3])
    customer_historical_otif = np.random.beta(8, 2, n_samples)  # Historical OTIF performance to this customer
    customer_order_frequency = np.random.choice(['Daily', 'Weekly', 'Monthly', 'Occasional'],
                                                 n_samples, p=[0.1, 0.3, 0.35, 0.25])
    
    # Fulfillment characteristics
    fulfillment_source = np.random.choice(['DC', 'Store', 'Vendor Direct', 'Cross-Dock'],
                                           n_samples, p=[0.45, 0.25, 0.15, 0.15])
    inventory_availability = np.random.beta(8, 2, n_samples)  # % of items available
    pick_complexity_score = np.random.uniform(0, 1, n_samples)
    
    # Logistics factors
    distance_miles = np.random.exponential(200, n_samples) + 10
    carrier_tiers = np.random.choice(['Premium', 'Standard', 'Economy'],
                                      n_samples, p=[0.2, 0.55, 0.25])
    carrier_performance_score = np.random.beta(7, 2, n_samples)
    
    # Timing factors
    order_day_of_week = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                          n_samples, p=[0.18, 0.17, 0.17, 0.17, 0.15, 0.1, 0.06])
    requested_delivery_window = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Anytime'],
                                                  n_samples, p=[0.15, 0.25, 0.15, 0.45])
    days_until_delivery = np.random.randint(1, 14, n_samples)
    
    # Operational factors
    warehouse_capacity_utilization = np.random.uniform(0.5, 1.0, n_samples)
    current_backlog_orders = np.random.poisson(50, n_samples)
    labor_availability_score = np.random.beta(7, 3, n_samples)
    
    # External factors  
    weather_risk_score = np.random.uniform(0, 1, n_samples)
    is_peak_season = np.random.binomial(1, 0.25, n_samples)
    
    # Generate OTIF risk score based on realistic relationships
    risk_score = (
        0.3  # base risk
        # Order complexity factors
        + 0.15 * (order_types == 'Same-Day').astype(float)
        + 0.1 * (order_types == 'Express').astype(float)
        + 0.1 * (order_size == 'Bulk').astype(float)
        + 0.05 * (order_size == 'Large').astype(float)
        + 0.02 * np.clip(n_line_items / 10, 0, 1)
        
        # Customer factors
        - 0.1 * (customer_tiers == 'Platinum').astype(float)  # Priority handling
        - 0.05 * (customer_tiers == 'Gold').astype(float)
        - 0.2 * customer_historical_otif  # Good history = less risk
        
        # Fulfillment factors
        - 0.25 * inventory_availability  # Full availability = lower risk
        + 0.15 * pick_complexity_score
        + 0.1 * (fulfillment_source == 'Vendor Direct').astype(float)  # Less control
        
        # Logistics factors
        + 0.001 * np.clip(distance_miles / 500, 0, 1)
        - 0.15 * carrier_performance_score
        + 0.1 * (carrier_tiers == 'Economy').astype(float)
        
        # Timing factors
        + 0.1 * (order_day_of_week == 'Friday').astype(float)
        + 0.15 * (order_day_of_week == 'Saturday').astype(float)
        + 0.1 * (requested_delivery_window == 'Morning').astype(float)
        - 0.05 * np.clip(days_until_delivery / 7, 0, 1)  # More lead time = less risk
        
        # Operational factors
        + 0.2 * (warehouse_capacity_utilization > 0.85).astype(float)
        + 0.01 * np.clip(current_backlog_orders / 100, 0, 1)
        - 0.15 * labor_availability_score
        
        # External factors
        + 0.2 * weather_risk_score
        + 0.15 * is_peak_season
        
        + np.random.normal(0, 0.08, n_samples)
    )
    
    # Convert to multi-class risk levels
    otif_risk = np.where(risk_score < 0.3, 0,  # Low Risk
                        np.where(risk_score < 0.55, 1, 2))  # Medium Risk / High Risk
    
    df = pd.DataFrame({
        'order_type': order_types,
        'order_size': order_size,
        'num_line_items': n_line_items,
        'order_value_usd': np.round(order_value_usd, 2),
        'customer_tier': customer_tiers,
        'customer_historical_otif_rate': np.round(customer_historical_otif, 3),
        'customer_order_frequency': customer_order_frequency,
        'fulfillment_source': fulfillment_source,
        'inventory_availability_rate': np.round(inventory_availability, 3),
        'pick_complexity_score': np.round(pick_complexity_score, 3),
        'distance_miles': np.round(distance_miles, 0).astype(int),
        'carrier_tier': carrier_tiers,
        'carrier_performance_score': np.round(carrier_performance_score, 3),
        'order_day_of_week': order_day_of_week,
        'requested_delivery_window': requested_delivery_window,
        'days_until_delivery': days_until_delivery,
        'warehouse_capacity_utilization': np.round(warehouse_capacity_utilization, 3),
        'current_backlog_orders': current_backlog_orders,
        'labor_availability_score': np.round(labor_availability_score, 3),
        'weather_risk_score': np.round(weather_risk_score, 3),
        'is_peak_season': is_peak_season,
        'otif_risk': otif_risk
    })
    
    return df


# =============================================================================
# Regression Datasets
# =============================================================================

def generate_price_elasticity_data(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Generate dataset for predicting price elasticity of demand.
    
    Features include product attributes, competitive pricing, and market conditions.
    Target: demand_change_pct (% change in demand for 1% price change)
    
    Use case: Demand Planning - Understand how price changes affect demand
    """
    np.random.seed(seed)
    
    # Product characteristics
    categories = np.random.choice(['Beverages', 'Snacks', 'Dairy', 'Frozen', 'Personal Care'],
                                   n_samples)
    is_private_label = np.random.binomial(1, 0.25, n_samples)
    brand_strength = np.random.beta(5, 2, n_samples)  # 0-1, higher = stronger brand
    
    # Pricing context
    current_price = np.random.uniform(2, 30, n_samples)
    price_tier = np.where(current_price < 8, 'Value',
                         np.where(current_price < 18, 'Mid', 'Premium'))
    competitor_price_ratio = np.random.uniform(0.7, 1.3, n_samples)  # our price / competitor price
    historical_promo_frequency = np.random.uniform(0, 0.5, n_samples)  # % time on promo
    
    # Market/demand characteristics
    category_growth_rate = np.random.uniform(-0.05, 0.15, n_samples)
    market_share = np.random.beta(2, 5, n_samples)
    purchase_frequency = np.random.choice(['Weekly', 'Monthly', 'Quarterly'], n_samples,
                                           p=[0.3, 0.5, 0.2])
    
    # Consumer demographics (aggregated at product level)
    avg_household_income = np.random.uniform(40000, 150000, n_samples)
    price_sensitivity_index = np.random.uniform(0.3, 0.9, n_samples)
    
    # Seasonality
    current_month = np.random.randint(1, 13, n_samples)
    
    # Calculate elasticity (target) - more negative = more elastic
    # Elasticity typically ranges from -4 (very elastic) to 0 (perfectly inelastic)
    base_elasticity = -1.2
    elasticity = (
        base_elasticity
        - 0.8 * (categories == 'Snacks').astype(float)  # snacks more elastic
        - 0.5 * (categories == 'Beverages').astype(float)
        + 0.6 * brand_strength  # strong brands less elastic
        + 0.4 * is_private_label  # private label more elastic
        - 0.3 * np.clip(competitor_price_ratio - 1, -0.3, 0.3)
        - 0.5 * historical_promo_frequency  # frequent promos = more elastic
        - 0.3 * price_sensitivity_index
        + 0.2 * (avg_household_income > 100000).astype(float)  # affluent less elastic
        + np.random.normal(0, 0.3, n_samples)
    )
    elasticity = np.clip(elasticity, -4.0, -0.1)
    
    df = pd.DataFrame({
        'category': categories,
        'is_private_label': is_private_label,
        'brand_strength_score': np.round(brand_strength, 3),
        'current_price_usd': np.round(current_price, 2),
        'price_tier': price_tier,
        'competitor_price_ratio': np.round(competitor_price_ratio, 3),
        'historical_promo_frequency': np.round(historical_promo_frequency, 3),
        'category_growth_rate': np.round(category_growth_rate, 3),
        'market_share': np.round(market_share, 3),
        'purchase_frequency': purchase_frequency,
        'avg_household_income': np.round(avg_household_income, 0).astype(int),
        'price_sensitivity_index': np.round(price_sensitivity_index, 3),
        'current_month': current_month,
        'price_elasticity': np.round(elasticity, 3)
    })
    
    return df


def generate_promotion_lift_data(n_samples: int = 2500, seed: int = 42) -> pd.DataFrame:
    """
    Generate dataset for predicting promotion lift (incremental sales).
    
    Features include promotion characteristics, product attributes, and timing.
    Target: promotion_lift_pct (% increase in sales during promotion)
    
    Use case: Demand Planning - Predict the sales impact of planned promotions
    """
    np.random.seed(seed)
    
    # Promotion characteristics
    promo_types = np.random.choice(['Price Discount', 'BOGO', 'Bundle', 'Display', 'Coupon'],
                                    n_samples, p=[0.35, 0.2, 0.15, 0.2, 0.1])
    discount_depth = np.random.uniform(0.05, 0.40, n_samples)  # % off
    promo_duration_days = np.random.choice([7, 14, 21, 28], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    has_display = np.random.binomial(1, 0.3, n_samples)
    has_feature_ad = np.random.binomial(1, 0.25, n_samples)
    
    # Product characteristics
    categories = np.random.choice(['Beverages', 'Snacks', 'Dairy', 'Frozen', 'Personal Care'],
                                   n_samples)
    base_price = np.random.uniform(3, 25, n_samples)
    brand_awareness = np.random.beta(5, 3, n_samples)
    is_new_product = np.random.binomial(1, 0.1, n_samples)
    
    # Historical performance
    baseline_units_weekly = np.random.exponential(100, n_samples)
    promo_frequency_annual = np.random.randint(0, 12, n_samples)
    last_promo_weeks_ago = np.random.randint(1, 52, n_samples)
    
    # Timing
    promo_week = np.random.randint(1, 53, n_samples)
    is_holiday_period = ((promo_week >= 46) | (promo_week <= 2) | 
                         ((promo_week >= 12) & (promo_week <= 16))).astype(int)
    
    # Competition
    competitor_promo_same_week = np.random.binomial(1, 0.3, n_samples)
    
    # Calculate promotion lift (target)
    base_lift = 50  # 50% base lift
    lift = (
        base_lift
        + 150 * discount_depth  # deeper discount = more lift
        + 40 * (promo_types == 'BOGO').astype(float)
        + 20 * (promo_types == 'Bundle').astype(float)
        + 35 * has_display
        + 25 * has_feature_ad
        + 15 * (categories == 'Snacks').astype(float)
        + 10 * (categories == 'Beverages').astype(float)
        + 30 * brand_awareness
        + 50 * is_new_product
        - 3 * promo_frequency_annual  # promo fatigue
        + 0.5 * last_promo_weeks_ago  # longer gap = more impact
        + 20 * is_holiday_period
        - 25 * competitor_promo_same_week
        - 0.5 * promo_duration_days  # longer promos = lower avg lift
        + np.random.normal(0, 15, n_samples)
    )
    lift = np.clip(lift, 5, 300)
    
    df = pd.DataFrame({
        'promotion_type': promo_types,
        'discount_depth_pct': np.round(discount_depth, 3),
        'promo_duration_days': promo_duration_days,
        'has_display': has_display,
        'has_feature_ad': has_feature_ad,
        'category': categories,
        'base_price_usd': np.round(base_price, 2),
        'brand_awareness_score': np.round(brand_awareness, 3),
        'is_new_product': is_new_product,
        'baseline_weekly_units': np.round(baseline_units_weekly, 0).astype(int),
        'promo_frequency_annual': promo_frequency_annual,
        'weeks_since_last_promo': last_promo_weeks_ago,
        'promo_week_of_year': promo_week,
        'is_holiday_period': is_holiday_period,
        'competitor_promo_same_week': competitor_promo_same_week,
        'promotion_lift_pct': np.round(lift, 1)
    })
    
    return df


def generate_supplier_lead_time_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate dataset for predicting actual supplier lead times.
    
    Features include supplier attributes, order characteristics, and logistics factors.
    Target: actual_lead_time_days
    
    Use case: Supply Planning - Predict actual delivery times for planning
    """
    np.random.seed(seed)
    
    # Supplier characteristics
    supplier_tiers = np.random.choice(['Strategic', 'Preferred', 'Approved', 'Conditional'],
                                       n_samples, p=[0.15, 0.3, 0.35, 0.2])
    supplier_regions = np.random.choice(['Domestic', 'LATAM', 'Europe', 'Asia'],
                                         n_samples, p=[0.35, 0.2, 0.2, 0.25])
    contracted_lead_time = np.random.randint(5, 60, n_samples)
    supplier_reliability_score = np.random.beta(7, 2, n_samples)
    
    # Order characteristics
    order_quantity = np.random.exponential(800, n_samples)
    order_complexity = np.random.choice(['Standard', 'Custom', 'Complex'],
                                         n_samples, p=[0.6, 0.3, 0.1])
    is_rush_order = np.random.binomial(1, 0.1, n_samples)
    
    # Logistics factors
    transport_mode = np.random.choice(['Ground', 'Ocean', 'Air', 'Rail'],
                                       n_samples, p=[0.4, 0.3, 0.2, 0.1])
    distance_miles = np.random.uniform(100, 8000, n_samples)
    port_of_entry = np.random.choice(['LA', 'Houston', 'Newark', 'Savannah', 'None'],
                                      n_samples, p=[0.2, 0.15, 0.15, 0.1, 0.4])
    
    # Timing factors
    order_month = np.random.randint(1, 13, n_samples)
    is_peak_logistics_season = ((order_month >= 8) & (order_month <= 11)).astype(int)
    
    # External factors
    customs_complexity = np.random.uniform(0, 1, n_samples)
    weather_disruption_prob = np.random.uniform(0, 0.3, n_samples)
    
    # Calculate actual lead time (target)
    base_delay = (
        contracted_lead_time
        + 5 * (supplier_tiers == 'Conditional').astype(float)
        + 3 * (supplier_tiers == 'Approved').astype(float)
        + 8 * (supplier_regions == 'Asia').astype(float)
        + 5 * (supplier_regions == 'Europe').astype(float)
        + 3 * (supplier_regions == 'LATAM').astype(float)
        - 8 * supplier_reliability_score
        + 2 * (order_complexity == 'Custom').astype(float)
        + 5 * (order_complexity == 'Complex').astype(float)
        - 3 * is_rush_order
        + 5 * (transport_mode == 'Ocean').astype(float)
        - 5 * (transport_mode == 'Air').astype(float)
        + 0.001 * distance_miles
        + 4 * is_peak_logistics_season
        + 5 * customs_complexity
        + 10 * weather_disruption_prob
        + np.random.normal(0, 3, n_samples)
    )
    actual_lead_time = np.clip(base_delay, contracted_lead_time * 0.8, contracted_lead_time * 2.5)
    actual_lead_time = np.round(actual_lead_time, 0).astype(int)
    
    df = pd.DataFrame({
        'supplier_tier': supplier_tiers,
        'supplier_region': supplier_regions,
        'contracted_lead_time_days': contracted_lead_time,
        'supplier_reliability_score': np.round(supplier_reliability_score, 3),
        'order_quantity_units': np.round(order_quantity, 0).astype(int),
        'order_complexity': order_complexity,
        'is_rush_order': is_rush_order,
        'transport_mode': transport_mode,
        'distance_miles': np.round(distance_miles, 0).astype(int),
        'port_of_entry': port_of_entry,
        'order_month': order_month,
        'is_peak_logistics_season': is_peak_logistics_season,
        'customs_complexity_score': np.round(customs_complexity, 3),
        'weather_disruption_probability': np.round(weather_disruption_prob, 3),
        'actual_lead_time_days': actual_lead_time
    })
    
    return df


def generate_yield_prediction_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate dataset for predicting production yield.
    
    Features include production parameters, equipment status, and material quality.
    Target: yield_percentage (0-100)
    
    Use case: Production Planning - Predict output yield for capacity planning
    """
    np.random.seed(seed)
    
    # Production characteristics
    product_lines = np.random.choice(['Line_A', 'Line_B', 'Line_C', 'Line_D'], n_samples)
    product_complexity = np.random.choice(['Simple', 'Standard', 'Complex'], n_samples,
                                           p=[0.3, 0.5, 0.2])
    batch_size = np.random.uniform(500, 5000, n_samples)
    
    # Equipment factors
    equipment_age_years = np.random.uniform(0.5, 15, n_samples)
    days_since_maintenance = np.random.randint(1, 90, n_samples)
    equipment_efficiency = np.random.beta(8, 2, n_samples)
    
    # Material quality
    raw_material_grade = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.3, 0.5, 0.2])
    material_moisture_pct = np.random.uniform(2, 12, n_samples)
    supplier_quality_score = np.random.beta(7, 2, n_samples)
    
    # Process parameters
    process_temperature = np.random.normal(180, 10, n_samples)
    process_pressure = np.random.normal(50, 5, n_samples)
    process_speed = np.random.uniform(80, 120, n_samples)  # % of standard
    
    # Operator factors
    shift = np.random.choice(['Day', 'Evening', 'Night'], n_samples, p=[0.4, 0.35, 0.25])
    operator_experience_years = np.random.exponential(5, n_samples)
    
    # Environmental
    ambient_temperature = np.random.normal(72, 8, n_samples)
    ambient_humidity = np.random.uniform(30, 70, n_samples)
    
    # Calculate yield (target)
    base_yield = 95
    yield_pct = (
        base_yield
        - 3 * (product_complexity == 'Complex').astype(float)
        - 1 * (product_complexity == 'Standard').astype(float)
        - 0.2 * equipment_age_years
        - 0.03 * days_since_maintenance
        + 5 * equipment_efficiency
        + 2 * (raw_material_grade == 'A').astype(float)
        - 2 * (raw_material_grade == 'C').astype(float)
        - 0.2 * np.abs(material_moisture_pct - 6)  # optimal at 6%
        + 3 * supplier_quality_score
        - 0.05 * np.abs(process_temperature - 180)
        - 0.1 * np.abs(process_pressure - 50)
        - 0.02 * np.abs(process_speed - 100)
        - 1 * (shift == 'Night').astype(float)
        + 0.1 * np.clip(operator_experience_years, 0, 20)
        - 0.05 * np.abs(ambient_temperature - 72)
        - 0.02 * np.abs(ambient_humidity - 50)
        + np.random.normal(0, 1.5, n_samples)
    )
    yield_pct = np.clip(yield_pct, 70, 99.5)
    
    df = pd.DataFrame({
        'production_line': product_lines,
        'product_complexity': product_complexity,
        'batch_size_units': np.round(batch_size, 0).astype(int),
        'equipment_age_years': np.round(equipment_age_years, 1),
        'days_since_maintenance': days_since_maintenance,
        'equipment_efficiency_score': np.round(equipment_efficiency, 3),
        'raw_material_grade': raw_material_grade,
        'material_moisture_pct': np.round(material_moisture_pct, 1),
        'supplier_quality_score': np.round(supplier_quality_score, 3),
        'process_temperature_f': np.round(process_temperature, 1),
        'process_pressure_psi': np.round(process_pressure, 1),
        'process_speed_pct': np.round(process_speed, 1),
        'shift': shift,
        'operator_experience_years': np.round(operator_experience_years, 1),
        'ambient_temperature_f': np.round(ambient_temperature, 1),
        'ambient_humidity_pct': np.round(ambient_humidity, 1),
        'yield_percentage': np.round(yield_pct, 2)
    })
    
    return df


def generate_transportation_lead_time_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate dataset for predicting transportation/delivery lead times.
    
    Features include shipment characteristics, route information, and logistics factors.
    Target: actual_transit_days
    
    Use case: Distribution Planning - Predict delivery times for logistics optimization
    """
    np.random.seed(seed)
    
    # Shipment characteristics
    shipment_types = np.random.choice(['Full Truckload', 'LTL', 'Parcel', 'Intermodal'],
                                       n_samples, p=[0.3, 0.35, 0.2, 0.15])
    weight_lbs = np.random.exponential(2000, n_samples) + 100
    volume_cubic_ft = weight_lbs / np.random.uniform(8, 15, n_samples)  # density varies
    n_pallets = np.ceil(volume_cubic_ft / 80).astype(int)  # ~80 cu ft per pallet
    is_hazmat = np.random.binomial(1, 0.05, n_samples)
    is_temperature_controlled = np.random.binomial(1, 0.15, n_samples)
    
    # Route information
    origin_regions = np.random.choice(['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest'],
                                       n_samples)
    dest_regions = np.random.choice(['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest'],
                                     n_samples)
    distance_miles = np.random.uniform(100, 3000, n_samples)
    n_stops = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.5, 0.25, 0.15, 0.07, 0.03])
    is_cross_border = np.random.binomial(1, 0.08, n_samples)
    
    # Carrier and service
    carrier_tiers = np.random.choice(['Premium', 'Standard', 'Economy'],
                                      n_samples, p=[0.2, 0.55, 0.25])
    carrier_on_time_rating = np.random.beta(8, 2, n_samples)
    service_level = np.random.choice(['Next Day', '2-Day', 'Ground', 'Economy'],
                                      n_samples, p=[0.1, 0.2, 0.5, 0.2])
    
    # Timing factors
    ship_day_of_week = np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
                                         n_samples, p=[0.2, 0.2, 0.2, 0.2, 0.15, 0.05])
    ship_month = np.random.randint(1, 13, n_samples)
    is_peak_shipping_season = ((ship_month >= 10) & (ship_month <= 12)).astype(int)
    
    # External factors
    fuel_surcharge_pct = np.random.uniform(0.05, 0.25, n_samples)
    weather_delay_probability = np.random.uniform(0, 0.3, n_samples)
    port_congestion_index = np.random.uniform(0, 1, n_samples) * is_cross_border
    
    # Calculate base transit time
    base_transit = (
        1.5  # minimum
        + distance_miles / 500  # ~500 miles per day average
        + 0.5 * (n_stops - 1)  # each stop adds time
        + 2 * is_cross_border
        + 1 * is_hazmat
        + 0.5 * is_temperature_controlled
    )
    
    # Adjust for service level and carrier
    service_multiplier = {'Next Day': 0.3, '2-Day': 0.5, 'Ground': 1.0, 'Economy': 1.3}
    carrier_multiplier = {'Premium': 0.85, 'Standard': 1.0, 'Economy': 1.2}
    
    actual_transit = (
        base_transit
        * np.array([service_multiplier[s] for s in service_level])
        * np.array([carrier_multiplier[c] for c in carrier_tiers])
        - 0.5 * carrier_on_time_rating
        + 0.5 * is_peak_shipping_season
        + 2 * weather_delay_probability
        + 1 * port_congestion_index
        + 0.3 * (ship_day_of_week == 'Fri').astype(float)
        + 0.5 * (ship_day_of_week == 'Sat').astype(float)
        + np.random.normal(0, 0.5, n_samples)
    )
    actual_transit = np.clip(actual_transit, 1, 15).round(0).astype(int)
    
    df = pd.DataFrame({
        'shipment_type': shipment_types,
        'weight_lbs': np.round(weight_lbs, 0).astype(int),
        'volume_cubic_ft': np.round(volume_cubic_ft, 1),
        'num_pallets': n_pallets,
        'is_hazmat': is_hazmat,
        'is_temperature_controlled': is_temperature_controlled,
        'origin_region': origin_regions,
        'destination_region': dest_regions,
        'distance_miles': np.round(distance_miles, 0).astype(int),
        'num_stops': n_stops,
        'is_cross_border': is_cross_border,
        'carrier_tier': carrier_tiers,
        'carrier_on_time_rating': np.round(carrier_on_time_rating, 3),
        'service_level': service_level,
        'ship_day_of_week': ship_day_of_week,
        'ship_month': ship_month,
        'is_peak_shipping_season': is_peak_shipping_season,
        'fuel_surcharge_pct': np.round(fuel_surcharge_pct, 3),
        'weather_delay_probability': np.round(weather_delay_probability, 3),
        'actual_transit_days': actual_transit
    })
    
    return df


# =============================================================================
# Anomaly Detection Datasets
# =============================================================================

def generate_scrap_anomaly_data(n_samples: int = 1000, anomaly_rate: float = 0.08,
                                 seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate dataset for detecting anomalous scrap/defect rates in production.
    
    Features include production metrics and quality indicators.
    Returns: (DataFrame, anomaly_labels) where 1 = anomaly
    
    Use case: Production Planning - Detect unusual scrap patterns indicating issues
    """
    np.random.seed(seed)
    
    n_normal = int(n_samples * (1 - anomaly_rate))
    n_anomaly = n_samples - n_normal
    
    # Normal production data
    normal_data = {
        'scrap_rate_pct': np.random.beta(2, 30, n_normal) * 100,  # typically 2-8%
        'defect_count': np.random.poisson(3, n_normal),
        'rework_hours': np.random.exponential(2, n_normal),
        'equipment_vibration': np.random.normal(50, 5, n_normal),
        'process_temperature_deviation': np.random.normal(0, 2, n_normal),
        'material_waste_pct': np.random.beta(2, 20, n_normal) * 100,
        'cycle_time_variance': np.random.normal(0, 3, n_normal),
        'operator_interventions': np.random.poisson(2, n_normal),
        'quality_score': np.random.beta(8, 2, n_normal) * 100,
        'downtime_minutes': np.random.exponential(10, n_normal),
    }
    
    # Anomalous production data (various anomaly patterns)
    anomaly_patterns = np.random.choice(['high_scrap', 'equipment', 'process', 'quality'],
                                         n_anomaly)
    
    anomaly_data = {
        'scrap_rate_pct': np.where(anomaly_patterns == 'high_scrap',
                                   np.random.uniform(15, 40, n_anomaly),
                                   np.random.beta(2, 30, n_anomaly) * 100 + 5),
        'defect_count': np.where(anomaly_patterns == 'quality',
                                 np.random.poisson(15, n_anomaly),
                                 np.random.poisson(5, n_anomaly)),
        'rework_hours': np.random.exponential(6, n_anomaly),
        'equipment_vibration': np.where(anomaly_patterns == 'equipment',
                                        np.random.normal(80, 10, n_anomaly),
                                        np.random.normal(55, 5, n_anomaly)),
        'process_temperature_deviation': np.where(anomaly_patterns == 'process',
                                                  np.random.normal(0, 10, n_anomaly),
                                                  np.random.normal(0, 3, n_anomaly)),
        'material_waste_pct': np.random.beta(2, 10, n_anomaly) * 100 + 3,
        'cycle_time_variance': np.random.normal(5, 5, n_anomaly),
        'operator_interventions': np.random.poisson(6, n_anomaly),
        'quality_score': np.random.beta(4, 4, n_anomaly) * 100,
        'downtime_minutes': np.random.exponential(30, n_anomaly),
    }
    
    # Combine data
    combined_data = {}
    for key in normal_data:
        combined_data[key] = np.concatenate([normal_data[key], anomaly_data[key]])
    
    # Add contextual features
    combined_data['production_line'] = np.random.choice(['Line_A', 'Line_B', 'Line_C'],
                                                         n_samples)
    combined_data['shift'] = np.random.choice(['Day', 'Evening', 'Night'], n_samples)
    combined_data['batch_number'] = np.arange(1, n_samples + 1)
    
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    for key in combined_data:
        if isinstance(combined_data[key], np.ndarray):
            combined_data[key] = combined_data[key][shuffle_idx]
    labels = labels[shuffle_idx]
    
    # Round numeric columns
    for key in ['scrap_rate_pct', 'rework_hours', 'equipment_vibration',
                'process_temperature_deviation', 'material_waste_pct',
                'cycle_time_variance', 'quality_score', 'downtime_minutes']:
        combined_data[key] = np.round(combined_data[key], 2)
    
    df = pd.DataFrame(combined_data)
    
    return df, labels.astype(int)


def generate_capacity_anomaly_data(n_samples: int = 1000, anomaly_rate: float = 0.1,
                                    seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate dataset for detecting capacity and downtime anomalies.
    
    Features include production metrics, equipment telemetry, and scheduling data.
    Returns: (DataFrame, anomaly_labels) where 1 = anomaly
    
    Use case: Production Planning - Detect unusual capacity patterns
    """
    np.random.seed(seed)
    
    n_normal = int(n_samples * (1 - anomaly_rate))
    n_anomaly = n_samples - n_normal
    
    # Normal data
    normal_data = {
        'capacity_utilization_pct': np.random.beta(8, 2, n_normal) * 100,  # ~80%
        'oee_score': np.random.beta(7, 3, n_normal) * 100,  # overall equipment effectiveness
        'planned_downtime_hours': np.random.exponential(2, n_normal),
        'unplanned_downtime_hours': np.random.exponential(0.5, n_normal),
        'changeover_time_hours': np.random.normal(1.5, 0.3, n_normal),
        'throughput_units_per_hour': np.random.normal(500, 50, n_normal),
        'setup_time_minutes': np.random.normal(45, 10, n_normal),
        'maintenance_alerts': np.random.poisson(1, n_normal),
        'energy_consumption_kwh': np.random.normal(1000, 100, n_normal),
        'labor_hours': np.random.normal(8, 1, n_normal),
    }
    
    # Anomaly data (unusual patterns)
    anomaly_data = {
        'capacity_utilization_pct': np.random.choice(
            [np.random.uniform(30, 50, n_anomaly),  # unexpectedly low
             np.random.uniform(95, 105, n_anomaly)],  # unusually high
        ),
        'oee_score': np.random.beta(3, 5, n_anomaly) * 100,
        'planned_downtime_hours': np.random.exponential(2, n_anomaly),
        'unplanned_downtime_hours': np.random.exponential(4, n_anomaly),  # much higher
        'changeover_time_hours': np.random.normal(3, 0.8, n_anomaly),  # double normal
        'throughput_units_per_hour': np.random.normal(350, 80, n_anomaly),  # lower
        'setup_time_minutes': np.random.normal(80, 20, n_anomaly),
        'maintenance_alerts': np.random.poisson(5, n_anomaly),
        'energy_consumption_kwh': np.random.normal(1300, 150, n_anomaly),
        'labor_hours': np.random.normal(10, 2, n_anomaly),
    }
    
    # If capacity_utilization is array of arrays, flatten
    if isinstance(anomaly_data['capacity_utilization_pct'], np.ndarray):
        if anomaly_data['capacity_utilization_pct'].ndim > 1:
            anomaly_data['capacity_utilization_pct'] = anomaly_data['capacity_utilization_pct'][0]
    
    # Combine
    combined_data = {}
    for key in normal_data:
        combined_data[key] = np.concatenate([normal_data[key], anomaly_data[key]])
    
    # Add context
    combined_data['production_line'] = np.random.choice(['Line_A', 'Line_B', 'Line_C', 'Line_D'],
                                                         n_samples)
    combined_data['day_of_week'] = np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                                                     n_samples)
    combined_data['shift'] = np.random.choice(['Day', 'Evening', 'Night'], n_samples)
    
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    for key in combined_data:
        if isinstance(combined_data[key], np.ndarray):
            combined_data[key] = combined_data[key][shuffle_idx]
    labels = labels[shuffle_idx]
    
    # Round
    for key in ['capacity_utilization_pct', 'oee_score', 'planned_downtime_hours',
                'unplanned_downtime_hours', 'changeover_time_hours', 'throughput_units_per_hour',
                'setup_time_minutes', 'energy_consumption_kwh', 'labor_hours']:
        combined_data[key] = np.round(combined_data[key], 2)
    
    df = pd.DataFrame(combined_data)
    
    return df, labels.astype(int)


# =============================================================================
# Time Series Forecasting Datasets
# =============================================================================

def generate_demand_forecast_data(n_skus: int = 100, n_stores: int = 50,
                                   n_weeks: int = 104, seed: int = 42) -> pd.DataFrame:
    """
    Generate time series demand data for forecasting.
    
    Features include SKU, store, date, and various demand drivers.
    Target: units_sold
    
    Use case: Demand Planning - Forecast product demand by SKU/location
    """
    np.random.seed(seed)
    
    # Sample subset of SKU-store combinations for tractability
    n_combinations = min(200, n_skus * n_stores)  # Limit for demo
    
    records = []
    start_date = datetime(2022, 1, 3)  # Start on a Monday
    
    # Pre-generate SKU and store characteristics
    sku_base_demand = {f"SKU{i:04d}": np.random.exponential(50) + 10 for i in range(n_skus)}
    sku_seasonality = {f"SKU{i:04d}": np.random.uniform(0.1, 0.5) for i in range(n_skus)}
    sku_trend = {f"SKU{i:04d}": np.random.uniform(-0.002, 0.005) for i in range(n_skus)}
    
    store_multiplier = {f"STR{i:04d}": np.random.uniform(0.5, 2.0) for i in range(n_stores)}
    
    # Generate combinations
    combinations = []
    for _ in range(n_combinations):
        sku = f"SKU{np.random.randint(0, n_skus):04d}"
        store = f"STR{np.random.randint(0, n_stores):04d}"
        combinations.append((sku, store))
    combinations = list(set(combinations))[:n_combinations]
    
    for sku, store in combinations:
        base = sku_base_demand[sku] * store_multiplier[store]
        seasonality = sku_seasonality[sku]
        trend = sku_trend[sku]
        
        for week in range(n_weeks):
            week_start = start_date + timedelta(weeks=week)
            
            # Seasonal component (annual cycle with Q4 peak)
            week_of_year = week_start.isocalendar()[1]
            seasonal_factor = 1 + seasonality * np.sin(2 * np.pi * (week_of_year - 13) / 52)
            
            # Trend component
            trend_factor = 1 + trend * week
            
            # Holiday effects
            is_holiday_week = week_of_year in [1, 22, 47, 48, 49, 50, 51, 52]
            holiday_lift = 1.3 if is_holiday_week else 1.0
            
            # Promotional effect (random promotions)
            is_on_promotion = np.random.random() < 0.15
            promo_lift = np.random.uniform(1.3, 1.8) if is_on_promotion else 1.0
            
            # Price changes
            price_index = np.random.uniform(0.9, 1.1)
            price_effect = 1 - 0.3 * (price_index - 1)  # simple elasticity
            
            # Calculate demand
            demand = (base * seasonal_factor * trend_factor * holiday_lift 
                     * promo_lift * price_effect * np.random.lognormal(0, 0.15))
            demand = max(0, int(round(demand)))
            
            records.append({
                'sku_id': sku,
                'store_id': store,
                'week_start_date': week_start,
                'week_of_year': week_of_year,
                'year': week_start.year,
                'is_holiday_week': int(is_holiday_week),
                'is_on_promotion': int(is_on_promotion),
                'price_index': round(price_index, 3),
                'units_sold': demand
            })
    
    df = pd.DataFrame(records)
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    
    return df


def generate_aggregate_demand_forecast_data(n_series: int = 50, n_months: int = 36,
                                             seed: int = 42) -> pd.DataFrame:
    """
    Generate aggregated monthly demand data (simpler format for quick demos).
    
    This is a simplified version suitable for the time series notebook demo.
    Target: demand_units
    """
    np.random.seed(seed)
    
    records = []
    start_date = datetime(2022, 1, 1)
    
    categories = ['Beverages', 'Snacks', 'Dairy', 'Frozen', 'Personal Care']
    regions = ['Northeast', 'Southeast', 'Midwest', 'West']
    
    for series_idx in range(n_series):
        # Series characteristics
        category = np.random.choice(categories)
        region = np.random.choice(regions)
        series_id = f"DEMAND_{series_idx:04d}"
        
        base_demand = np.random.uniform(5000, 50000)
        seasonality_amp = base_demand * np.random.uniform(0.15, 0.4)
        trend = np.random.uniform(-50, 200)  # monthly trend
        
        # Category-specific seasonal phase
        seasonal_phase = {'Beverages': 3, 'Snacks': 0, 'Dairy': -1, 
                         'Frozen': 4, 'Personal Care': -2}[category]
        
        for month_idx in range(n_months):
            date = start_date + pd.DateOffset(months=month_idx)
            
            # Seasonal (yearly cycle)
            month = date.month
            seasonal = seasonality_amp * np.sin(2 * np.pi * (month - seasonal_phase) / 12)
            
            # Trend
            trend_value = trend * month_idx
            
            # Holiday lift (Nov-Dec)
            holiday_factor = 1.25 if month in [11, 12] else 1.0
            
            # Random noise
            noise = np.random.normal(0, base_demand * 0.08)
            
            demand = max(0, (base_demand + seasonal + trend_value) * holiday_factor + noise)
            
            records.append({
                'series_id': series_id,
                'category': category,
                'region': region,
                'date': date,
                'year': date.year,
                'month': month,
                'demand_units': int(round(demand))
            })
    
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    
    return df


# =============================================================================
# Helper Functions
# =============================================================================

def encode_categorical_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    One-hot encode categorical columns for ML models.
    
    Returns:
        (encoded_df, encoding_info_dict)
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        return df.copy(), {}
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    encoding_info = {
        'original_columns': categorical_cols,
        'encoded_columns': [c for c in df_encoded.columns if c not in df.columns 
                           or c in categorical_cols]
    }
    
    return df_encoded, encoding_info


def prepare_features_target(df: pd.DataFrame, target_col: str,
                            exclude_cols: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare feature matrix X and target vector y from DataFrame.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        exclude_cols: Columns to exclude from features
    
    Returns:
        (X, y, feature_names)
    """
    exclude = set(exclude_cols or [])
    exclude.add(target_col)
    
    # Get numeric columns for features
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # Encode categoricals
    df_subset = df[feature_cols + [target_col]].copy()
    df_encoded, _ = encode_categorical_columns(df_subset)
    
    # Get final feature columns (excluding target)
    final_features = [c for c in df_encoded.columns if c != target_col]
    
    X = df_encoded[final_features].values
    y = df_encoded[target_col].values
    
    return X, y, final_features


def create_lag_features(series: np.ndarray, n_lags: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lag features for time series forecasting.
    
    Args:
        series: 1D array of time series values
        n_lags: Number of lag features to create
    
    Returns:
        (X, y) where X has shape (n_samples, n_lags)
    """
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


# =============================================================================
# Demo/Test
# =============================================================================

if __name__ == "__main__":
    print("Generating sample datasets...\n")
    
    # Classification
    print("1. Supplier Delay Risk Data:")
    df_delay = generate_supplier_delay_risk_data(n_samples=100)
    print(f"   Shape: {df_delay.shape}")
    print(f"   Target distribution: {df_delay['is_delayed'].value_counts().to_dict()}\n")
    
    print("2. Material Shortage Data:")
    df_shortage = generate_material_shortage_data(n_samples=100)
    print(f"   Shape: {df_shortage.shape}")
    print(f"   Target distribution: {df_shortage['shortage_risk'].value_counts().to_dict()}\n")
    
    print("3. Labor Shortage Data:")
    df_labor = generate_labor_shortage_data(n_samples=100)
    print(f"   Shape: {df_labor.shape}")
    print(f"   Target distribution: {df_labor['labor_shortage_risk'].value_counts().to_dict()}\n")
    
    print("4. OTIF Risk Data:")
    df_otif = generate_otif_risk_data(n_samples=100)
    print(f"   Shape: {df_otif.shape}")
    print(f"   Target distribution: {df_otif['otif_risk'].value_counts().to_dict()}\n")
    
    # Regression
    print("5. Price Elasticity Data:")
    df_elasticity = generate_price_elasticity_data(n_samples=100)
    print(f"   Shape: {df_elasticity.shape}")
    print(f"   Target range: [{df_elasticity['price_elasticity'].min():.2f}, {df_elasticity['price_elasticity'].max():.2f}]\n")
    
    print("6. Promotion Lift Data:")
    df_promo = generate_promotion_lift_data(n_samples=100)
    print(f"   Shape: {df_promo.shape}")
    print(f"   Target range: [{df_promo['promotion_lift_pct'].min():.1f}%, {df_promo['promotion_lift_pct'].max():.1f}%]\n")
    
    print("7. Supplier Lead Time Data:")
    df_supplier_lt = generate_supplier_lead_time_data(n_samples=100)
    print(f"   Shape: {df_supplier_lt.shape}")
    print(f"   Target range: [{df_supplier_lt['actual_lead_time_days'].min()}, {df_supplier_lt['actual_lead_time_days'].max()}] days\n")
    
    print("8. Transportation Lead Time Data:")
    df_transport_lt = generate_transportation_lead_time_data(n_samples=100)
    print(f"   Shape: {df_transport_lt.shape}")
    print(f"   Target range: [{df_transport_lt['actual_transit_days'].min()}, {df_transport_lt['actual_transit_days'].max()}] days\n")
    
    print("9. Yield Prediction Data:")
    df_yield = generate_yield_prediction_data(n_samples=100)
    print(f"   Shape: {df_yield.shape}")
    print(f"   Target range: [{df_yield['yield_percentage'].min():.1f}%, {df_yield['yield_percentage'].max():.1f}%]\n")
    
    # Anomaly Detection
    print("10. Scrap Anomaly Data:")
    df_scrap, labels_scrap = generate_scrap_anomaly_data(n_samples=100)
    print(f"   Shape: {df_scrap.shape}")
    print(f"   Anomaly rate: {labels_scrap.mean():.1%}\n")
    
    # Time Series
    print("11. Demand Forecast Data:")
    df_demand = generate_aggregate_demand_forecast_data(n_series=5, n_months=24)
    print(f"   Shape: {df_demand.shape}")
    print(f"   Series: {df_demand['series_id'].nunique()}")
    
    print("\nAll datasets generated successfully!")
