from datetime import timedelta
from typing import Tuple, Optional, Dict, List
import pandas as pd
from utils import stderr_print
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain import hub

pd.set_option('display.max_columns', None)

department_names = [
    "Electronics", "Home Goods", "Apparel", "Groceries", "Pharmacy", "Automotive",
    "Sporting Goods", "Toys", "Garden Center", "Jewelry", "Books & Media", "Health & Beauty",
    "Pets", "Hardware", "Crafts", "Party Supplies", "Stationery", "Seasonal",
    "Bakery", "Deli", "Produce", "Meat & Seafood", "Dairy & Frozen", "Snacks & Drinks",
    "Baby Products", "Personal Care", "Household Ess.", "Cleaning Supplies", "Paper Goods",
    "Floral", "Footwear", "Luggage", "Outdoor Living", "Camping Gear", "Fishing Gear",
    "Hunting Gear", "Fitness Equip.", "Team Sports", "Video Games", "Movies & TV",
    "Music", "Computers", "Cell Phones", "Tablets", "Wearable Tech", "TVs",
    "Audio Equip.", "Cameras", "Office Furn.", "School Supplies", "Art Supplies",
    "Fabric", "Sewing Notions", "Yarn & Needlework", "Scrapbooking", "Gift Cards",
    "Gift Wrap", "Balloons", "Greeting Cards", "Candles", "Home Decor", "Kitchenware",
    "Bedding", "Bath", "Storage Org.", "Lighting", "Small App.", "Large App.",
    "Tires", "Auto Parts", "Motor Oil", "Car Care", "Tools", "Power Tools",
    "Hand Tools", "Hardware Acc.", "Plumbing", "Electrical", "Paint", "Flooring",
    "Building Mats.", "Lawn Care", "Pest Control", "Pool & Spa", "Patio Furn.",
    "Grills", "Bird Food", "Pet Food", "Pet Toys", "Pet Beds", "Fish & Aquatics",
    "Reptile Supp.", "Small Anim. Supp.", "Hunting Lic.", "Fishing Lic.", "Career Services",
    "Eye Clinic", "Soda"
]

def filter_stores(df: pd.DataFrame, storeID: List[int]) -> pd.DataFrame:
    if storeID:
        valid_stores = [s for s in storeID if 1 <= s <= 45]
        if valid_stores:
            df = df[df['Store'].isin(valid_stores)]
        else:
            stderr_print("no valid stores in range [1,45]")
    else:
        stderr_print("all stores")
    return df

def get_department_name(dept_id) -> str:
    """
    gets a department name from the list, error handling for invalid IDs.
    :param dept_id: id of the department, as-is
    :return: dept name
    """
    if (type(dept_id) == str): return dept_id
    if pd.isna(dept_id):
        return 'Unknown Dept'
    index = int(dept_id) - 1

    if 0 <= index < len(department_names):
        return str(department_names[index])
    else:
        return 'Invalid Dept'

def _week_start(first_dt: pd.Timestamp, dt: pd.Timestamp) -> pd.Timestamp:
    """
    calculate start date of the week of a given date
    the first week begins at the earliest date
    :param first_dt: first date in the entire dataset.
    :param dt: date to find the corresponding week start for.
    :return: The timestamp representing the start of the week.
    """
    days_since_start = (dt - first_dt).days
    week_number = days_since_start // 7
    return first_dt + timedelta(days=week_number * 7)

def add_all_data(sales_df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """
    reads entire data and merges it with a given dataframe. only matching dates are included.
    @:param sales_df: sales dataframe
    @:return gen_df, containing store, dept, date, w_sales & isHoliday;
    spec_df containing store, date, w_sales, temp, fuel_price, cpi, unemployment
    """
    features_df = pd.read_csv(csv_path, usecols=['Store', 'Date', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'])
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    merged_df = pd.merge(sales_df, features_df, on=['Store', 'Date'], how='inner')
    final_df = merged_df[['Store', 'Date', 'Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
    return final_df

def parse_sales_csv(
        train_csv_path: str, features_csv_path: str = './test_data/features.csv', chunksize: int = 50000,
        storeID: List[int] = [1]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    read data from csv in chunks, add to the df
    :param train_csv_path: path of csv file w data
    :param features_csv_path: path of csv file w features
    :param chunksize: size of chunk to read from csv
    :param storeID: id of stores to include
    :return:
    """
    reader = pd.read_csv(train_csv_path, chunksize=chunksize, iterator=True, encoding='unicode_escape')
    chunks = []
    weekly_sales_agg: Dict[int, List[float]] = {}
    for i in range(1, 46): weekly_sales_agg[i] = []
    for chunk in reader:
        store_id, sales_col = 'Store', 'Weekly_Sales'
        chunk[store_id] = pd.to_numeric(chunk[store_id])
        chunk[sales_col] = pd.to_numeric(chunk[sales_col])
        for store, sales in zip(chunk[store_id], chunk[sales_col]):
            weekly_sales_agg[store].append(sales)
        chunks.append(chunk)
    general_df = pd.concat(chunks, ignore_index=True)
    spec_df = add_all_data(general_df, features_csv_path)
    if storeID:
        valid_stores = [s for s in storeID if 1 <= s <= 45]
        if valid_stores:
            stderr_print(f"for stores {valid_stores}")
            general_df = general_df[general_df['Store'].isin(valid_stores)]
            spec_df = spec_df[spec_df['Store'].isin(valid_stores)]
        else:
            stderr_print("no valid stores in range [1,45]")
    else:
        stderr_print("all stores")
    return general_df, spec_df


@tool
def get_total_sales_for_stores_for_years(gen_df: pd.DataFrame, storeIDs: List[int] = [1], years: List[int] = [2010]) -> int:
    """
    Calculates the sum of all sales for the given years for the given stores
    :param gen_df: DataFrame to sum with
    :param storeIDs: List of store IDs to sum for, defaults to [1]
    :param years: List of years to sum for, defaults to [2010]
    :return: sales as int
    """
    gen_df = gen_df.drop_duplicates('Store')['Weekly_Sales']
    

@tool
def get_total_sales_for_stores_for_months(gen_df: pd.DataFrame, storeIDs: List[int] = [1], years: List[int] = [2010]) -> int:
    """
    Calculates the sum of all sales for the given years for the given stores
    :param gen_df: DataFrame to sum with
    :param storeIDs: List of store IDs to sum for, defaults to [1]
    :param years: List of years to sum for, defaults to [2010]
    :return: sales as int
    """
    return gen_df.drop_duplicates('Store')['Weekly_Sales'].sum()

@tool
def get_total_sales_for_stores_for_dates(gen_df: pd.DataFrame, storeIDs: List[int] = [1], years: List[int] = [2010]) -> int:
    """
    Calculates the sum of all sales for the given years for the given stores
    :param gen_df: DataFrame to sum with
    :param storeIDs: List of store IDs to sum for, defaults to [1]
    :param years: List of years to sum for, defaults to [2010]
    :return: sales as int
    """
    return gen_df.drop_duplicates('Store')['Weekly_Sales'].sum()

gen_df, specialized_df = (parse_sales_csv("./test_data/train.csv"))
# print(gen_df)
# print(specialized_df)
stderr_print("Gen df:")
stderr_print(gen_df)
stderr_print("Spec df:")
stderr_print(specialized_df)