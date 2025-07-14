from typing import List
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from crud import parse_sales_csv, get_department_name
from utils import stderr_print

def generate_graphs(gen_df: pd.DataFrame, spec_df: pd.DataFrame, isPred: int = 0, storeID: List[int] = None) -> str:
    stderr_print("generating eda")
    result: str = ""

    # Filtering gen and spec DFs

    if storeID:
        valid_stores = [s for s in storeID if 1 <= s <= 45]
        if valid_stores:
            stderr_print(f"for stores {valid_stores}")
            result += f"For Stores {valid_stores}:"
            gen_df = gen_df[gen_df['Store'].isin(valid_stores)]
            spec_df = spec_df[spec_df['Store'].isin(valid_stores)]
        else:
            stderr_print("no valid stores in range [1,45]")
    else:
        stderr_print("all stores")
    spec_df['Date'] = pd.to_datetime(spec_df['Date'])
    spec_df.sort_values('Date', inplace=True)
    spec_df.drop_duplicates(subset=['Date'], inplace=True)
    spec_df = spec_df.set_index('Date')
    stderr_print("Received from parse_sales_csv, gen & spec, filtered:")
    stderr_print(gen_df)
    stderr_print(spec_df)

    # Generating all features

    result += __sales_t(gen_df)
    if (not isPred): result += __holiday_impact(gen_df)
    if (storeID == -1): result += __top_performers(gen_df)
    # result += __unemployment_correlation(spec_df)
    if (not isPred): result += __department_analysis(gen_df, isHoliday=1)
    if (not isPred): result += __get_propensity_score(spec_df)
    if storeID != -1: result += f"These statistics are only for Store {storeID}."
    return result

def __sales_t(df: pd.DataFrame, isForecasted: int = 0) -> str:
    """
    choosing plotly over pyplotlib as it looks better for larger datasets
    :param df: General DataFrame to iterate over
    """
    stderr_print("generating sales graph")
    title = "(Forecasted) Total Weekly Sales Over Time" if isForecasted else "Total Weekly Sales Over Time"
    time_series = df.groupby('Date', as_index=False)['Weekly_Sales'].sum()
    fig = px.line(
        time_series,
        x='Date',
        y='Weekly_Sales',
        title=title,
        labels={'Weekly_Sales': 'Total Weekly Sales ($)'}
    )
    fig.show()

    peak_sales_week = time_series.loc[time_series['Weekly_Sales'].idxmax()]
    lowest_sales_week = time_series.loc[time_series['Weekly_Sales'].idxmin()]
    average_sales = time_series['Weekly_Sales'].mean()

    summary = (
        f"Overall sales peaked at ${peak_sales_week['Weekly_Sales']:,.0f} on the week of {peak_sales_week['Date']}. "
        f"The lowest point was ${lowest_sales_week['Weekly_Sales']:,.0f} on {lowest_sales_week['Date']}. "
        f"The average weekly sales across the entire period was ${average_sales:,.0f}.\n"
    )
    return summary

def __holiday_impact(df: pd.DataFrame) -> str:
    stderr_print("generating holiday impact box plot")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='IsHoliday', y='Weekly_Sales', data=df)
    plt.xlabel("Is Holiday")
    plt.ylabel("Weekly Sales ($)")
    plt.title("Distribution of Weekly Sales: Holiday vs Non-Holiday")
    plt.grid(True, axis='y')
    plt.show()

    holiday_sales = df[df['IsHoliday'] == True]['Weekly_Sales'].mean()
    non_holiday_sales = df[df['IsHoliday'] == False]['Weekly_Sales'].mean()
    percentage_diff = ((holiday_sales - non_holiday_sales) / non_holiday_sales) * 100

    summary = (
        f"On average, sales during holiday weeks (${holiday_sales:,.0f}) were {percentage_diff:.1f}% higher "
        f"than sales during non-holiday weeks (${non_holiday_sales:,.0f}).\n"
    )
    return summary

def __top_performers(df: pd.DataFrame, top_n: int = 5, isForecasted: int = 0) -> str:
    stderr_print("generating top performers bar graph")
    sales_by_store = df.groupby('Store', as_index=False)['Weekly_Sales'].sum()
    sales_by_store = sales_by_store.sort_values(by='Weekly_Sales', ascending=False)

    title = f"(Forecasted) Top {top_n} Performing Stores by Total Sales" if isForecasted else f"Top {top_n} Performing Stores by Total Sales"
    fig = px.bar(
        sales_by_store.head(top_n),
        x='Store',
        y='Weekly_Sales',
        title=title,
        labels={'Store': 'Store ID', 'Weekly_Sales': 'Total Sales ($)'},
        text_auto='.2s'
    )
    fig.show()

    top_store = sales_by_store.iloc[0]
    bottom_store = sales_by_store.iloc[-1]

    summary = (
        f"Store {top_store['Store']} was the top performer with total sales of ${top_store['Weekly_Sales']:,.0f}. "
        f"In contrast, Store {bottom_store['Store']} had the lowest sales with a total of ${bottom_store['Weekly_Sales']:,.0f}.\n"
    )
    return summary

def __unemployment_correlation(spec_df: pd.DataFrame) -> str:
    """
    Generate correlation heatmap between Weekly_Sales and Unemployment for each store
    :param spec_df: DataFrame with columns Store, Date, Weekly_Sales, Unemployment
    """
    stderr_print("generating unemployment correlation heatmap")

    correlations = []
    for store in spec_df['Store'].unique():
        store_data = spec_df[spec_df['Store'] == store]
        if len(store_data) > 1:  # Need at least 2 data points for correlation
            corr = store_data['Weekly_Sales'].corr(store_data['Unemployment'])
            correlations.append({'Store': store, 'Correlation': corr})

    corr_df = pd.DataFrame(correlations)

    heatmap_data = corr_df.set_index('Store')[['Correlation']].T

    plt.figure(figsize=(25, 5))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap='RdBu_r',
        center=0,
        fmt='.3f',
        cbar_kws={'label': 'Correlation Coefficient'},
        annot_kws={'size': 8}
    )
    plt.title("Correlation between Weekly Sales and Unemployment by Store")
    plt.xlabel("Store ID")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.show()

    avg_correlation = corr_df['Correlation'].mean()
    strongest_positive = corr_df.loc[corr_df['Correlation'].idxmax()]
    strongest_negative = corr_df.loc[corr_df['Correlation'].idxmin()]

    summary = (
        f"The average correlation between weekly sales and unemployment across all stores is {avg_correlation:.3f}. "
        f"Store {strongest_positive['Store']} shows the strongest positive correlation ({strongest_positive['Correlation']:.3f}), "
        f"while Store {strongest_negative['Store']} has the strongest negative correlation ({strongest_negative['Correlation']:.3f}).\n"
    )

    return summary

def __department_analysis(spec_df: pd.DataFrame, isHoliday: int = 0, top_n: int = 5) -> str:
    print(f"\n\n\n{spec_df.head()}\n\n\n")
    spec_df['Dept'] = spec_df['Dept'].apply(get_department_name)
    if isHoliday: spec_df = spec_df[spec_df['IsHoliday'] == True]
    sales_by_dept = spec_df.groupby('Dept', as_index=False)['Weekly_Sales'].sum()
    sales_by_dept = sales_by_dept.sort_values(by='Weekly_Sales', ascending=False)

    fig = px.bar(
        sales_by_dept.head(top_n),
        x='Dept',
        y='Weekly_Sales',
        title=f"Top {top_n} Performing Depts by Total Sales",
        labels={'Dept': 'Dept ID', 'Weekly_Sales': 'Total Sales ($)'},
        text_auto='.2s'
    )
    fig.show()

    top_dept = sales_by_dept.iloc[0]
    bottom_dept = sales_by_dept.iloc[-1]

    summary = (
        f"Dept {top_dept['Dept']} was the top performer with total sales of ${top_dept['Weekly_Sales']:,.0f}. "
        f"In contrast, Dept {bottom_dept['Dept']} had the lowest sales with a total of ${bottom_dept['Weekly_Sales']:,.0f}.\n"
    )
    return summary

def __get_propensity_score(spec_df: pd.DataFrame) -> str:
    """
    Creating a propensity score for measuring externa; factors against performance. Here, forecasting is avoided since
    it'll be hard to predict external factors of the market with only these paremeters. However, this may serve towards
    1. Comparing with weekly sales to examine responses to external factors
    2. Is the budget flexible enough to react to changes?
    :param spec_df: DF with date, CPI, unemployment, fuel_price and temperature, arranged weekly
    :return: None
    """

    # ideal temperature in Celsius. We measure the distance from this.
    # Since we want all factors to be proportional for growth, we invert this score.

    IDEAL_TEMPERATURE: int = 21
    FEATURES = ['Temperature', 'CPI', 'Unemployment', 'Fuel_Price']
    spec_df['Temperature'] = -1 * abs(IDEAL_TEMPERATURE - spec_df['Temperature'])
    sc = StandardScaler()
    spec_df_features = spec_df[FEATURES]

    # Creating standardized Z-Scores for features

    print(spec_df.head())
    spec_df_scaled = pd.DataFrame(sc.fit_transform(spec_df_features), columns=FEATURES, index=spec_df.index)
    sc.fit(spec_df_features)

    # Using PCA to find the optimal weights

    pca = PCA(n_components=1)
    pca.fit(spec_df_scaled)
    weights = pca.components_[0]

    # Creating an index value

    unemployment_weight = pd.Series(weights, index=FEATURES)['Unemployment']
    if unemployment_weight > 0:
        weights = -weights
    spec_df['headwinds_index'] = spec_df_scaled.dot(weights)

    # Scale the index

    spec_df['headwinds_index'] = spec_df['headwinds_index'] * 10
    # max_val = spec_df['headwinds_index'].max()
    # spec_df['headwinds_index_scaled'] = 100 * (spec_df['headwinds_index'] - min_val) / (max_val - min_val)
    # stderr_print(spec_df[['headwinds_index_scaled'] + FEATURES])

    # Visual

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(spec_df.index, spec_df['headwinds_index'], label='Retail Headwinds Index', color='b', lw=2)
    ax.set_title('Weekly Retail Headwinds Index (2-Year Period)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Index Score (Higher = More Favorable)', fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.show()

    # Generating features to measure trend, seasonality and lag

    WINDOW_SHORT = 12 # Approx. 1 Quarter
    WINDOW_LONG = 52  # Approx. 1 Year

    # Statistical Summary

    spec_df['rolling_avg'] = spec_df['headwinds_index'].rolling(window=WINDOW_SHORT, min_periods=1).mean()
    spec_df['volatility'] = spec_df['headwinds_index'].rolling(window=WINDOW_SHORT, min_periods=1).std()

    # Trend

    spec_df['yearly_trend'] = spec_df['headwinds_index'].pct_change(periods=WINDOW_LONG) * 100

    # Lag

    spec_df['lag_4_weeks'] = spec_df['headwinds_index'].shift(4)
    spec_df['lag_12_weeks'] = spec_df['headwinds_index'].shift(12)

    # Generating textual summary

    latest_data = spec_df.iloc[-1]

    # Generate overall statistics

    overall_avg = spec_df['headwinds_index'].mean()
    overall_volatility = spec_df['volatility'].mean()
    max_index_val = spec_df['headwinds_index'].max()
    min_index_val = spec_df['headwinds_index'].min()
    date_of_max = spec_df['headwinds_index'].idxmax().strftime('%B %d, %Y')
    date_of_min = spec_df['headwinds_index'].idxmin().strftime('%B %d, %Y')

    summary = [f"The most recent Retail Headwinds Index is {latest_data['headwinds_index']:.2f}."]

    # Compare the latest index to its short-term rolling average

    if latest_data['headwinds_index'] > latest_data['rolling_avg'] * 1.05:
        summary.append("This is significantly above the recent 12-week average, indicating strong favorable conditions.")
    elif latest_data['headwinds_index'] < latest_data['rolling_avg'] * 0.95:
        summary.append("This is significantly below the recent 12-week average, indicating challenging headwinds.")
    else:
        summary.append("This is in line with the recent 12-week average.")

    # Add long-term context

    summary.append(
        f"Over the entire period, the index averaged {overall_avg:.2f}, with a peak of "
        f"{max_index_val:.2f} on {date_of_max} and a low of {min_index_val:.2f} on {date_of_min}."
    )

    # Analyze trends

    latest_yoy_trend = latest_data['yearly_trend']
    if pd.notna(latest_yoy_trend):
        if latest_yoy_trend > 5:
            trend_desc = f"a strong positive year-over-year trend, improving by {latest_yoy_trend:.1f}%"
        elif latest_yoy_trend < -5:
            trend_desc = f"a notable negative year-over-year trend, worsening by {abs(latest_yoy_trend):.1f}%"
        else:
            trend_desc = "a stable year-over-year trend"
        summary.append(f"The most recent data shows {trend_desc}.")
    else:
        summary.append("A full year of data is not yet available to determine the year-over-year trend.")

    # Analyze volatility

    latest_volatility = latest_data['volatility']
    if pd.notna(latest_volatility) and pd.notna(overall_volatility):
        volatility_comparison = "similar to"
        if latest_volatility > overall_volatility * 1.2:
            volatility_comparison = "higher than"
        elif latest_volatility < overall_volatility * 0.8:
            volatility_comparison = "lower than"

        summary.append(
            f"Recent market stability, measured by volatility over the last quarter ({latest_volatility:.2f}), "
            f"is {volatility_comparison} the period's average of {overall_volatility:.2f}."
        )

    return " ".join(summary)

general_df, specialized_df = (parse_sales_csv("./test_data/train.csv", storeID=[1]))
print(generate_graphs(general_df, storeID=[1], spec_df=specialized_df))
# __get_propensity_score(specialized_df)