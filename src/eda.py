from typing import List
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from crud import parse_sales_csv, get_department_name
from utils import stderr_print

def generate_graphs(gen_df: pd.DataFrame, spec_df: pd.DataFrame, isPred: int = 0, storeID: List[int] = None) -> str:
    stderr_print("generating eda")
    stderr_print(gen_df)
    result: str = ""
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
    result += __sales_t(gen_df)
    if (not isPred): result += __holiday_impact(gen_df)
    if (storeID == -1): result += __top_performers(gen_df)
    result += __unemployment_correlation(spec_df)
    if (not isPred): result += __department_analysis(gen_df, isHoliday=1)
    if storeID != -1: result += f"These statistics are only for Store {storeID}."
    return result

def __sales_t(df: pd.DataFrame) -> str:
    """
    choosing plotly over pyplotlib as it looks better for larger datasets
    :param df: General DataFrame to iterate over
    """
    stderr_print("generating sales graph")
    time_series = df.groupby('Date', as_index=False)['Weekly_Sales'].sum()
    fig = px.line(
        time_series,
        x='Date',
        y='Weekly_Sales',
        title="Total Weekly Sales Over Time",
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

def __top_performers(df: pd.DataFrame, top_n: int = 5) -> str:
    stderr_print("generating top performers bar graph")
    sales_by_store = df.groupby('Store', as_index=False)['Weekly_Sales'].sum()
    sales_by_store = sales_by_store.sort_values(by='Weekly_Sales', ascending=False)

    fig = px.bar(
        sales_by_store.head(top_n),
        x='Store',
        y='Weekly_Sales',
        title=f"Top {top_n} Performing Stores by Total Sales",
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

general_df, specialized_df = (parse_sales_csv("../test_data/train.csv"))
print(generate_graphs(general_df, storeID=[15,17,21,23,25], spec_df=specialized_df))