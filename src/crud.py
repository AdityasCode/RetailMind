import json
import os
from typing import Optional, Dict, List, Any
import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from src.utils import print_stderr, sales_attribute, filter_stores, default_storeIDs
from langchain_core.tools import StructuredTool
pd.set_option('display.max_columns', None)
class CRUD:
    def __init__(self, sales_path: str = "./test_data/train.csv", features_path: str = "./test_data/features.csv",
                 daily_path: str = "./test_data/train_daily_1.csv", storeIDs: List[int] = default_storeIDs):
        self.gen_df: pd.DataFrame = pd.DataFrame()
        self.spec_df: pd.DataFrame = pd.DataFrame()
        self.daily_df: pd.DataFrame = pd.DataFrame()
        self.storeIDs: List[int] = storeIDs
        self.add_sales_data(sales_path)
        self.add_external_factors(features_path)
        self.add_daily_data(daily_path)

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
        print_stderr("adding external factors")
        sales_df = self.gen_df
        print_stderr(sales_df)
        features_df = pd.read_csv(csv_path, usecols=['Store', 'Date', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'])
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        merged_df = pd.merge(sales_df, features_df, on=['Store', 'Date'], how='inner')
        final_df = merged_df[['Store', 'Date', 'Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
        self.spec_df = filter_stores(final_df, self.storeIDs)

    def add_sales_data(self, train_csv_path: str, chunksize: int = 50000) -> None:
        """
        read data from csv in chunks, add to the df
        :param train_csv_path: path of csv file w data
        :param features_csv_path: path of csv file w features
        :param chunksize: size of chunk to read from csv
        :param storeID: id of stores to include
        :return: None
        """
        print_stderr("adding sales data")
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
        self.gen_df = filter_stores(general_df, self.storeIDs)

    def get_total_sales_for_stores_for_years(self, storeIDs: Optional[List[int]] = None, years: Optional[List[int]] = None) -> int:
        if self.gen_df.empty: return -1
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
        if self.gen_df.empty: return -1
        if storeIDs is None:
            storeIDs = self.storeIDs
        if months is None:
            months = [2] # Defaults to February

        filtered_df = self.gen_df[
            (self.gen_df['Store'].isin(storeIDs)) &
            (self.gen_df['Date'].dt.month.isin(months))
            ]
        return int(filtered_df['Weekly_Sales'].sum())

    def get_total_sales_for_stores_for_dates(self, storeIDs: Optional[List[int]] = None, dates: Optional[List[str]] = None, isDaily: bool = False) -> int:
        if self.gen_df.empty: return -1
        attrUse, colUse = sales_attribute(isDaily=isDaily)
        if storeIDs is None:
            storeIDs = self.storeIDs
        defaultDateSet = [pd.to_datetime("2010-02-05")]
        target_dates = pd.to_datetime(dates)
        if target_dates is None or len(target_dates) == 0:
            target_dates = defaultDateSet
        targetDf: pd.DataFrame = self.daily_df if isDaily else self.gen_df
        filtered_df = targetDf[
            (targetDf['Store'].isin(storeIDs)) &
            (targetDf['Date'].isin(target_dates))
            ]
        if filtered_df.empty:
            filtered_df = targetDf[
                (targetDf['Store'].isin(storeIDs)) &
                (targetDf['Date'].isin(defaultDateSet))
                ]
        return int(filtered_df[colUse].sum())

    def add_daily_data(self, path: str, chunksize: int = 50000) -> pd.DataFrame:
        print_stderr("adding daily sales data")
        print_stderr(f"cwd here: {os.getcwd()}")
        reader = pd.read_csv(path, chunksize=chunksize, iterator=True, encoding='unicode_escape')
        chunks = []
        daily_sales_agg: Dict[int, List[float]] = {}
        for i in range(1, 46): daily_sales_agg[i] = []
        for chunk in reader:
            store_id, sales_col = 'Store', 'Daily_Sales'
            chunk[store_id] = pd.to_numeric(chunk[store_id])
            chunk[sales_col] = pd.to_numeric(chunk[sales_col])
            for store, sales in zip(chunk[store_id], chunk[sales_col]):
                daily_sales_agg[store].append(sales)
            chunks.append(chunk)
        general_df = pd.concat(chunks, ignore_index=True)
        self.daily_df = general_df

class Event:
    """
    Object for storing information about a specific event, such as name, description, start and end date.
    """
    def __init__(self, name: str, start_date: str, end_date: str, text: str, embedding: List[float] = None):
        """
        Initialize Event. Embedding is initialized when added to a Log object.
        :param name: name of event
        :param start_date: start date, can be string
        :param end_date: end date, can be string
        :param text: text
        :param embedding: leave unfulfilled or None if initializing event standalone
        """
        self.description = name
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.text = text
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Event object to a JSON-serializable dictionary
        :return: dict
        """
        return {
            "description": self.description,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "text": self.text,
            "embedding": self.embedding
        }

class EventLog:
    """
    Manages a collection of Event objects, including loading, saving, and vector searching.
    """
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.log: List[Event] = []
        self.model = SentenceTransformer(embedding_model_name)

    def add_event_from_text(self, description: str, start_date: str, end_date: str, event_text: str):
        """
        Deprecated
        """
        embedding = self.model.encode(event_text).tolist()
        event = Event(description, start_date, end_date, event_text, embedding)
        self.log.append(event)
        print_stderr(f"Event '{description}' added to log.")

    def add_event_from_event(self, event: Event):
        event.embedding = self.model.encode(event.description).tolist()
        self.log.append(event)
        print_stderr(f"Event '{event.description}' added to log.")

    def find_event(self, query: str) -> Event | None:
        """Finds the most relevant event in the log using vector similarity search."""
        if not self.log:
            return None

        query_embedding = self.model.encode(query)

        # Calculate cosine similarity between the query and all event embeddings
        similarities = [1 - cosine(query_embedding, event.embedding) for event in self.log]

        # Find the index of the most similar event
        most_similar_index = similarities.index(max(similarities))

        return self.log[most_similar_index]

    def save_log(self, path: str = "./events_log.json"):
        """Saves the entire event log to a JSON file."""
        with open(path, 'w') as f:
            json.dump([event.to_dict() for event in self.log], f, indent=4)
        print_stderr(f"Event log saved to {path}")

    def load_log(self, path: str = "./events_log.json"):
        """Loads events from a JSON file."""
        if not os.path.exists(path):
            print_stderr("No event log file found.")
            return

        with open(path, 'r') as f:
            events_data = json.load(f)
            self.log = [Event(**data) for data in events_data]
        print_stderr(f"Event log loaded from {path}")

    def toString(self):
        t = "Event log:"
        for event in self.log:
            t += event.description + "\n"
        return t