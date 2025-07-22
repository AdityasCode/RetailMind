from datetime import timedelta
from typing import Tuple, Optional, Dict, List
import pandas as pd
from utils import stderr_print
from langchain_core.tools import StructuredTool

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

class CRUD:
    def __init__(self, sales_path: str = "./test_data/train.csv", features_path: str = "./test_data/features.csv",
                 storeIDs: List[int] = [1]):
        self.gen_df: pd.DataFrame = pd.DataFrame()
        self.spec_df: pd.DataFrame = pd.DataFrame()
        self.add_sales_data(sales_path)
        self.add_external_factors(features_path)
        self.storeIDs: List[int] = storeIDs

        self.sales_by_year_tool = StructuredTool.from_function(
            func=self.get_total_sales_for_stores_for_years,
            name="get_sales_for_years",
            description="Calculates the sum of all sales for the given years for the given stores."
        )
        self.sales_by_month_tool = StructuredTool.from_function(
            func=self.get_total_sales_for_stores_for_months,
            name="get_sales_for_months",
            description="Calculates the sum of all sales for the given months for the given stores."
        )
        self.sales_by_date_tool = StructuredTool.from_function(
            func=self.get_total_sales_for_stores_for_dates,
            name="get_sales_for_dates",
            description="Calculates the sum of all sales for the given dates for the given stores."
        )


    def add_external_factors(self, csv_path: str) -> None:
        """
        reads entire data and merges it with a given dataframe. only matching dates are included.
        :param sales_df: sales dataframe
        :return None
        """
        stderr_print("adding external factors")
        sales_df = self.gen_df
        stderr_print(sales_df)
        features_df = pd.read_csv(csv_path, usecols=['Store', 'Date', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'])
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        merged_df = pd.merge(sales_df, features_df, on=['Store', 'Date'], how='inner')
        final_df = merged_df[['Store', 'Date', 'Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
        self.spec_df = final_df

    def add_sales_data(self, train_csv_path: str, chunksize: int = 50000) -> None:
        """
        read data from csv in chunks, add to the df
        :param train_csv_path: path of csv file w data
        :param features_csv_path: path of csv file w features
        :param chunksize: size of chunk to read from csv
        :param storeID: id of stores to include
        :return: None
        """
        stderr_print("adding sales data")
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
        self.gen_df = general_df

    def parse_sales_data(self,
                       storeID: List[int] = [1]
                       ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Returns both gen and spec dataframes
        :param storeID: list of store IDs to include || -1
        :return: gen_df, containing store, dept, date, w_sales & isHoliday;
        spec_df containing store, date, w_sales, temp, fuel_price, cpi, unemployment
        """
        return filter_stores(self.gen_df, storeID), filter_stores(self.get_spec_df(), storeID)


    def get_total_sales_for_stores_for_years(self, storeIDs: Optional[List[int]] = None, years: Optional[List[int]] = None) -> int:
        if storeIDs is None:
            storeIDs = self.storeIDs
        if years is None:
            years = [2010]

        filtered_df = self.gen_df[
            (self.gen_df['Store'].isin(storeIDs)) &
            (self.gen_df['Date'].dt.year.isin(years))
            ]
        return int(filtered_df['Weekly_Sales'].sum())

    def get_total_sales_for_stores_for_months(self, storeIDs: Optional[List[int]] = None, months: Optional[List[int]] = None) -> int:
        if storeIDs is None:
            storeIDs = self.storeIDs
        if months is None:
            months = [2] # Defaults to February

        filtered_df = self.gen_df[
            (self.gen_df['Store'].isin(storeIDs)) &
            (self.gen_df['Date'].dt.month.isin(months))
            ]
        return int(filtered_df['Weekly_Sales'].sum())

    def get_total_sales_for_stores_for_dates(self, storeIDs: Optional[List[int]] = None, dates: Optional[List[str]] = None) -> int:
        if storeIDs is None:
            storeIDs = self.storeIDs
        if dates is None:
            dates = ["2010-02-05"]

        target_dates = pd.to_datetime(dates)
        filtered_df = self.gen_df[
            (self.gen_df['Store'].isin(storeIDs)) &
            (self.gen_df['Date'].isin(target_dates))
            ]
        return int(filtered_df['Weekly_Sales'].sum())

# crudder = CRUD()
# stderr_print(crudder.gen_df)
# stderr_print(crudder.spec_df)