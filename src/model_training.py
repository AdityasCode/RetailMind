import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import pickle
from typing import Tuple
from scipy.linalg import pinv
from src.utils import print_stderr

warnings.filterwarnings('ignore')

def hierarchical_forecast_with_reconciliation(
        gen_df: pd.DataFrame,
        forecast_periods: int = 12,
        output_dir: str = "../charts",
        model_dir: str = "../models"
) -> Tuple[pd.DataFrame, str]:
    """
    Generate hierarchical forecasts with MinT reconciliation for retail sales data.
    Hierarchy: Total -> Store -> Department
    we choose a summing matrix so that calculations are precise and forecasts for the lowest level
    (department-store) sum up to total forecasted company sales.
    We then use an identity matrix as covariance matrix W as weights for trust of each individual forecast
    And use the mint formula to create a multiplier for all forecasts to minimize error variance
    This is better than top-down and bottom-up because both approaches ignore important trends.
    Further, MinT minimizes total forecast uncertainty.

    :param gen_df: DataFrame
    :param forecast_periods: Number of weeks to forecast
    :param output_dir: Directory to save charts
    :param model_dir: Directory to save models, enter None for no saving

    :returns (reconciled_forecasts_df, summary_text)
    """

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if model_dir is not None:
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    df = gen_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Dept', 'Date'])

    print_stderr("Building hierarchy structure...")

    # Data preparation: create hierarchy levels (just str ids)
    # L2 (Bottom): Store-Department combinations -> L1 (Middle): Store totals
    # -> L0 (Top): Total sales for company

    level2_data = df.groupby(['Store', 'Dept', 'Date'])['Weekly_Sales'].sum().reset_index()
    level2_data['series_id'] = level2_data['Store'].astype(str) + '_' + level2_data['Dept'].astype(str)
    level1_data = df.groupby(['Store', 'Date'])['Weekly_Sales'].sum().reset_index()
    level1_data['series_id'] = 'Store_' + level1_data['Store'].astype(str)
    level0_data = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    level0_data['series_id'] = 'Total'
    level0_data['Store'] = 0

    print_stderr(f"Hierarchy built: 1 total + {len(level1_data['Store'].unique())} stores + {len(level2_data['series_id'].unique())} store-dept combinations")

    all_forecasts = {}
    trained_models = {}

    # Try to find a model

    model_path = Path(model_dir) / "hierarchical_models.pkl"
    models_loaded = False
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                trained_models = pickle.load(f)
            print_stderr(f"Loaded {len(trained_models)} existing models from {model_path}")
            models_loaded = True
        except Exception as e:
            print_stderr(f"Failed to load existing models: {e}. Training new models...")
            models_loaded = False

    def train_and_forecast(data, series_id, periods=forecast_periods):
        """
        train Prophet model and generate forecast
        """
        series_data = data[data['series_id'] == series_id].copy()
        if len(series_data) < 10:  # skips if too few data points
            return None, None

        # Prophet requires ds as datestamp and y as the target

        prophet_data = series_data[['Date', 'Weekly_Sales']].rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})

        # adds holiday indicators for prophet's holiday fnality

        if 'IsHoliday' in df.columns:
            holiday_data = df[df['IsHoliday'] == True]['Date'].unique()
            holidays = pd.DataFrame({
                'holiday': 'retail_holiday',
                'ds': pd.to_datetime(holiday_data),
                'lower_window': 0,
                'upper_window': 0,
            })
        else:
            holidays = None

        # Train

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays,
            seasonality_mode='multiplicative'
        )

        model.fit(prophet_data)
        forecast_only = generate_forecast_from_model(model, periods)
        forecast_only['series_id'] = series_id

        return forecast_only, model

    def generate_forecast_from_model(model, periods=forecast_periods):
        """
        Generate forecast from existing trained model
        """

        # Generate future dates and extract only the forecast period

        future = model.make_future_dataframe(periods=periods, freq='W')
        forecast = model.predict(future)
        forecast_only = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        return forecast_only

    # Prepare all series that need forecasting

    all_series = (
            level0_data['series_id'].unique().tolist() +
            level1_data['series_id'].unique().tolist() +
            level2_data['series_id'].unique().tolist()
    )

    if models_loaded:
        print_stderr("Generating forecasts from loaded models...")
        for series_id in all_series:
            if series_id in trained_models:
                try:
                    forecast = generate_forecast_from_model(trained_models[series_id], forecast_periods)
                    forecast['series_id'] = series_id
                    all_forecasts[series_id] = forecast
                except Exception as e:
                    print_stderr(f"Failed to generate forecast for {series_id}: {e}")
    else:
        print_stderr("Training models and generating base forecasts...")
        all_data = pd.concat([
            level0_data[['series_id', 'Date', 'Weekly_Sales']],
            level1_data[['series_id', 'Date', 'Weekly_Sales']],
            level2_data[['series_id', 'Date', 'Weekly_Sales']]
        ])

        for series_id in all_series:
            forecast, model = train_and_forecast(all_data, series_id, forecast_periods)
            if forecast is not None:
                all_forecasts[series_id] = forecast
                trained_models[series_id] = model

        print_stderr(f"Successfully trained {len(trained_models)} models")

        # Save models

        if model_dir is not None:
            with open(model_path, 'wb') as f:
                pickle.dump(trained_models, f)
            print_stderr(f"Models saved to {model_path}")

    # Create reconciliation matrices

    print_stderr("Applying MinT reconciliation...")

    # Build summing matrix S, create mapping; Summing matrix: [Total, Stores, Bottom] -> [Bottom]

    n_total = 1
    n_stores = len(level1_data['Store'].unique())
    n_bottom = len(level2_data['series_id'].unique()) # depts
    store_mapping = {store: i for i, store in enumerate(sorted(level1_data['Store'].unique()))}
    bottom_mapping = {series_id: i for i, series_id in enumerate(sorted(level2_data['series_id'].unique()))}
    n_all = n_total + n_stores + n_bottom
    S = np.zeros((n_all, n_bottom))
    S[n_total + n_stores:, :] = np.eye(n_bottom) # Bottom level maps to itself

    # Store level: sum departments within each store, total level: sum all

    for i, (series_id, row_idx) in enumerate(bottom_mapping.items()):
        store = int(series_id.split('_')[0])
        if store in store_mapping:
            store_idx = store_mapping[store]
            S[n_total + store_idx, row_idx] = 1
    S[0, :] = 1

    # Collect and fill base forecasts in matrix form

    base_forecasts = np.zeros((n_all, forecast_periods))
    if 'Total' in all_forecasts:
        base_forecasts[0, :] = all_forecasts['Total']['yhat'].values

    for store, store_idx in store_mapping.items():
        store_series_id = f'Store_{store}'
        if store_series_id in all_forecasts:
            base_forecasts[n_total + store_idx, :] = all_forecasts[store_series_id]['yhat'].values

    for series_id, bottom_idx in bottom_mapping.items():
        if series_id in all_forecasts:
            base_forecasts[n_total + n_stores + bottom_idx, :] = all_forecasts[series_id]['yhat'].values

    # MinT reconciliation
    # For simplicity, use identity covariance (can be improved with residual covariance estimation)
    # but this might require more data
    # Reconciliation formula: S * (S' * W^-1 * S)^-1 * S' * W^-1

    W = np.eye(n_all)
    try:
        SWS_inv = pinv(S.T @ np.linalg.inv(W) @ S)
        G = S @ SWS_inv @ S.T @ np.linalg.inv(W)
        reconciled_forecasts = G @ base_forecasts
    except:

        # Fallback to bottom-up if MinT fails

        print_stderr("MinT reconciliation failed, using bottom-up approach")
        reconciled_forecasts = S @ base_forecasts[n_total + n_stores:, :]

    # Create reconciled forecast DataFrame

    print_stderr("Creating output dataframes and visualizations...")

    # Get future dates

    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=forecast_periods, freq='W')

    reconciled_df_list = []

    # Total forecasts

    for t, date in enumerate(future_dates):
        reconciled_df_list.append({
            'Date': date,
            'Store': 'Total',
            'Dept': 'All',
            'series_id': 'Total',
            'Weekly_Sales': reconciled_forecasts[0, t],
            'level': 'Total'
        })

    # Store forecasts

    for store, store_idx in store_mapping.items():
        for t, date in enumerate(future_dates):
            reconciled_df_list.append({
                'Date': date,
                'Store': store,
                'Dept': 'All',
                'series_id': f'Store_{store}',
                'Weekly_Sales': reconciled_forecasts[n_total + store_idx, t],
                'level': 'Store'
            })

    # Department forecasts

    for series_id, bottom_idx in bottom_mapping.items():
        store, dept = series_id.split('_')
        for t, date in enumerate(future_dates):
            reconciled_df_list.append({
                'Date': date,
                'Store': int(store),
                'Dept': int(dept),
                'series_id': series_id,
                'Weekly_Sales': reconciled_forecasts[n_total + n_stores + bottom_idx, t],
                'level': 'Department'
            })

    reconciled_df = pd.DataFrame(reconciled_df_list)

    # Create visualizations

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    # Plot 1: Total sales forecast

    total_history = level0_data.tail(52)  # Last year of data
    total_forecast = reconciled_df[reconciled_df['series_id'] == 'Total']

    axes[0, 0].plot(total_history['Date'], total_history['Weekly_Sales'], 'b-', label='Historical', linewidth=2)
    axes[0, 0].plot(total_forecast['Date'], total_forecast['Weekly_Sales'], 'r--', label='Forecast', linewidth=2)
    axes[0, 0].set_title('Total Sales Forecast', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Weekly Sales ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Top 5 stores forecast

    store_forecasts = reconciled_df[reconciled_df['level'] == 'Store']
    top_stores = (store_forecasts.groupby('Store')['Weekly_Sales'].sum()
                  .sort_values(ascending=False).head(5).index)

    for store in top_stores:
        store_data = store_forecasts[store_forecasts['Store'] == store]
        axes[0, 1].plot(store_data['Date'], store_data['Weekly_Sales'],
                        marker='o', label=f'Store {store}', linewidth=2, markersize=4)

    axes[0, 1].set_title('Top 5 Stores Forecast', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Weekly Sales ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Department performance heatmap

    dept_summary = (reconciled_df[reconciled_df['level'] == 'Department']
                    .groupby(['Store', 'Dept'])['Weekly_Sales'].mean().reset_index())
    dept_pivot = dept_summary.pivot(index='Store', columns='Dept', values='Weekly_Sales')

    # Select top departments by total sales

    top_depts = (dept_summary.groupby('Dept')['Weekly_Sales'].sum()
                 .sort_values(ascending=False).head(20).index)
    dept_pivot_subset = dept_pivot[top_depts]

    sns.heatmap(dept_pivot_subset, ax=axes[1, 0], cmap='YlOrRd',
                cbar_kws={'label': 'Avg Weekly Sales ($)'})
    axes[1, 0].set_title('Department Performance Heatmap (Top 20 Depts)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Department')
    axes[1, 0].set_ylabel('Store')

    # Plot 4: Forecast growth trends
    # Compare forecasted growth vs historical average

    historical_avg = df.groupby(['Store', 'Dept'])['Weekly_Sales'].mean().reset_index()
    forecast_avg = (reconciled_df[reconciled_df['level'] == 'Department']
                    .groupby(['Store', 'Dept'])['Weekly_Sales'].mean().reset_index())

    growth_comparison = historical_avg.merge(forecast_avg, on=['Store', 'Dept'], suffixes=('_hist', '_forecast'))
    growth_comparison['growth_rate'] = ((growth_comparison['Weekly_Sales_forecast'] -
                                         growth_comparison['Weekly_Sales_hist']) /
                                        growth_comparison['Weekly_Sales_hist'] * 100)

    # Top growing and declining departments

    top_growth = growth_comparison.nlargest(10, 'growth_rate')
    top_decline = growth_comparison.nsmallest(10, 'growth_rate')

    y_pos = np.arange(len(top_growth))
    axes[1, 1].barh(y_pos, top_growth['growth_rate'], color='green', alpha=0.7)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([f"S{row['Store']}-D{row['Dept']}" for _, row in top_growth.iterrows()])
    axes[1, 1].set_xlabel('Growth Rate (%)')
    axes[1, 1].set_title('Top 10 Growing Store-Department Combinations', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = Path(output_dir) / "hierarchical_forecast_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Generate summary insights

    total_forecast_sum = reconciled_df[reconciled_df['series_id'] == 'Total']['Weekly_Sales'].sum()
    historical_total = df['Weekly_Sales'].sum()
    periods_in_history = len(df['Date'].unique())
    avg_historical_weekly = historical_total / periods_in_history

    top_store = (reconciled_df[reconciled_df['level'] == 'Store']
    .groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False).index[0])

    top_dept = (reconciled_df[reconciled_df['level'] == 'Department']
    .groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False).index[0])

    summary_text = f"""
HIERARCHICAL FORECAST SUMMARY ({forecast_periods} weeks ahead)

KEY INSIGHTS:
• Total forecasted sales: ${total_forecast_sum:,.0f} over {forecast_periods} weeks
• Average weekly forecast: ${total_forecast_sum/forecast_periods:,.0f} vs historical ${avg_historical_weekly:,.0f}
• Growth outlook: {((total_forecast_sum/forecast_periods)/avg_historical_weekly-1)*100:.1f}% change from historical average

TOP PERFORMERS:
• Highest forecasted store: Store {top_store}
• Best performing department: Department {top_dept}
• Total stores analyzed: {len(store_mapping)} stores
• Total departments analyzed: {len(bottom_mapping)} store-department combinations

FORECAST HIGHLIGHTS:
• Models trained with hierarchical reconciliation (MinT method)
• Forecasts are mathematically coherent (store totals sum to company total)
• {len(trained_models)} individual time series models trained and reconciled
• Seasonal patterns and holiday effects incorporated

GROWTH OPPORTUNITIES:
• Fastest growing combination: Store {top_growth.iloc[0]['Store']} - Dept {top_growth.iloc[0]['Dept']} (+{top_growth.iloc[0]['growth_rate']:.1f}%)
• Areas needing attention: {len(growth_comparison[growth_comparison['growth_rate'] < 0])} store-dept combinations showing decline

TECHNICAL NOTES:
• Reconciliation method: Minimum Trace (MinT) 
• Base model: Prophet with seasonality and holiday effects
• Hierarchy levels: Total → Store → Department
• Chart saved to: {chart_path}
"""

    print_stderr("Hierarchical forecasting completed successfully!")
    print_stderr(f"Chart saved to: {chart_path}")

    return reconciled_df, summary_text
