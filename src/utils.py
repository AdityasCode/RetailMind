from datetime import timedelta
from sys import stderr
from typing import Tuple, List, Any

import fitz
import pandas as pd

default_storeIDs: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def print_stderr(text) -> None:
    """
    print to stderr instead of stdout
    :param text: text to print
    :return:
    """
    print(text, file=stderr)

def sales_attribute(isDaily: bool) -> Tuple[str, str]:
    f = 'Weekly Sales' if not isDaily else 'Daily Sales'
    b = f.replace(' ', '_')
    return f, b


def filter_stores(df: pd.DataFrame, storeID: List[int]) -> pd.DataFrame:
    if storeID:
        valid_stores = [s for s in storeID if 1 <= s <= 45]
        if valid_stores:
            df = df[df['Store'].isin(valid_stores)]
        else:
            print_stderr("no valid stores in range [1,45]")
    else:
        print_stderr("all stores")
    return df

def get_department_id(dept_name) -> int:
    if dept_name:
        if type(dept_name) == int: return dept_name
        try:
            tmp = department_names.index(dept_name)
            if type(tmp) != int:
                return -1
            return tmp
        except ValueError:
            return -1

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

def parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


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
