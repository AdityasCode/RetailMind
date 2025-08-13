import os.path
from typing import List, Tuple
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from langchain_core.tools import StructuredTool
from src.crud import EventLog
from src.model_training import hierarchical_forecast_with_reconciliation
from src.utils import sales_attribute, filter_stores, get_department_name, get_department_id, default_storeIDs
from src.utils import print_stderr
from pathlib import Path

class EDAFeatures:
    """
    EDA Features to analyze data.
    """
    def __init__(self, gen_df: pd.DataFrame, spec_df: pd.DataFrame, event_log: EventLog, predictor: TimeSeriesPredictor | None,
                 daily_df: pd.DataFrame = None, storeIDs: List[int] = default_storeIDs):
        """
        Initializes the toolkit with dataframes. Gen df and spec df must have the given columns respectively, case-sensitive:
        Store | Dept | Date | Weekly_Sales | IsHoliday
        Store | Date | Weekly_Sales | Temperature | Fuel_Price | CPI | Unemployment
        """
        self.gen_df = gen_df
        self.spec_df = spec_df
        self.daily_df = daily_df
        self.log = event_log
        self.storeIDs = storeIDs
        self.predictor = predictor

        # Filtering gen and spec DFs

        self.gen_df = filter_stores(self.gen_df, storeIDs)
        self.spec_df = filter_stores(self.spec_df, storeIDs)
        self.spec_df['Date'] = pd.to_datetime(self.spec_df['Date'])
        self.spec_df.sort_values('Date', inplace=True)
        self.spec_df.drop_duplicates(subset=['Date'], inplace=True)
        self.spec_df = self.spec_df.set_index('Date')
        # store col can be dropped from spec_df
        print_stderr("Received from parse_sales_csv, gen & spec, filtered:")
        gen_df, spec_df = self.gen_df, self.spec_df
        print_stderr(gen_df)
        print_stderr(spec_df)
        self.pred_df: pd.DataFrame = pd.DataFrame()

        self.output_dir = Path("../charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sales_t_tool = StructuredTool.from_function(
            func=self.sales_t,
            name="sales_over_time_chart",
            description="Generates a graph and summary for sales vs. time."
        )
        self.holiday_impact_tool = StructuredTool.from_function(
            func=self.holiday_impact,
            name="holiday_impact_analyzer",
            description="Creates a box plot and summary for the holiday impact on sales."
        )
        self.top_performing_stores_tool = StructuredTool.from_function(
            func=self.top_performing_stores,
            name="top_performing_stores_analyzer",
            description="Analyzes the top performing stores by total sales."
        )
        self.department_analysis_holiday_tool = StructuredTool.from_function(
            func=self.department_analysis_holiday,
            name="department_analysis_holiday",
            description="Analyzes department performance during holiday weeks."
        )
        self.department_analysis_no_holiday_tool = StructuredTool.from_function(
            func=self.department_analysis_no_holiday,
            name="department_analysis_non_holiday",
            description="Analyzes department performance during non-holiday weeks."
        )
        self.analyze_economic_headwinds_tool = StructuredTool.from_function(
            func=self.analyze_economic_headwinds,
            name="economic_headwinds_analyzer",
            description="Analyzes the impact of economic factors like CPI and unemployment on sales."
        )
        self.analyze_event_impact_tool = StructuredTool.from_function(
            func=self.analyze_event_impact,
            name="event_impact_analyzer",
            description="Analyzes the impact of an event on sales."
        )
        self.forecast_sales_tool = StructuredTool.from_function(
            func=self.forecast_weekly_sales,
            name="forecast_weekly_sales",
            description="Generates a sales forecast for a specified number of future weeks."
        )

    def get_gen_df(self) -> pd.DataFrame: return self.gen_df
    def get_spec_df(self) -> pd.DataFrame: return self.spec_df
    def set_gen_df(self, gen_df: pd.DataFrame) -> None: self.gen_df = gen_df
    def set_spec_df(self, spec_df: pd.DataFrame) -> None: self.spec_df = spec_df

    def generate_graphs(self) -> str:
        print_stderr("generating eda")
        result: str = ""

        # Generating all features

        result += self.sales_t_tool.invoke({})
        result += self.holiday_impact_tool.invoke({})
        if self.storeIDs and len(self.storeIDs) > 1: result += self.top_performing_stores_tool.invoke({})
        result += self.department_analysis_holiday_tool.invoke({})
        result += self.department_analysis_no_holiday_tool.invoke({})
        result += self.analyze_economic_headwinds_tool.invoke({})
        if self.storeIDs: result += f"These statistics are only for Store(s) {self.storeIDs}."
        return result

    def sales_t(self) -> str:
        """
        Generates a graph for sales vs. time.
        :return: summary with full file path for chart display.
        """
        attrUse, colUse = sales_attribute(isDaily=False)
        print_stderr("generating sales graph. gen_df:")
        print_stderr(self.gen_df)
        targetDf: pd.DataFrame = self.gen_df
        time_series = targetDf.groupby('Date', as_index=False)[colUse].sum()
        chart_path = self.output_dir / "sales.png"
        if not os.path.exists(chart_path):
            title = f"Total {attrUse} Over Time"
            fig = px.line(
                time_series,
                x='Date',
                y=colUse,
                title=title,
                labels={colUse: f'Total {attrUse} ($)'}
            )
            fig.write_image(chart_path)

        peak_sales_period = time_series.loc[time_series[colUse].idxmax()]
        lowest_sales_period = time_series.loc[time_series[colUse].idxmin()]
        average_sales = time_series[colUse].mean()

        summary = (
            f"Overall sales peaked at ${peak_sales_period[colUse]:,.0f} on the week of {peak_sales_period['Date']}. "
            f"The lowest point was ${lowest_sales_period[colUse]:,.0f} on {lowest_sales_period['Date']}. "
            f"The average {attrUse} across the entire period was ${average_sales:,.0f}.\n"
            f"Chart saved to: {chart_path}"
        )
        return summary

    def holiday_impact(self) -> str:
        """
        Creates a box plot for the holiday impact on sales.
        :return: summary with full file path for chart display.
        """
        attrUse, colUse = sales_attribute(isDaily=False)
        targetDf: pd.DataFrame = self.gen_df

        chart_path = self.output_dir / "holiday.png"
        print_stderr("generating holiday impact box plot")
        if (not os.path.exists(chart_path)):
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='IsHoliday', y=colUse, data=targetDf)
            plt.xlabel("Is Holiday")
            plt.ylabel(f"{attrUse} ($)")
            plt.title(f"Distribution of {attrUse}: Holiday vs Non-Holiday")
            plt.grid(True, axis='y')
            plt.savefig(chart_path)

        holiday_sales = targetDf[targetDf['IsHoliday'] == True][colUse].mean()
        non_holiday_sales = targetDf[targetDf['IsHoliday'] == False][colUse].mean()
        percentage_diff = ((holiday_sales - non_holiday_sales) / non_holiday_sales) * 100

        summary = (
            f"On average, sales during holiday weeks (${holiday_sales:,.0f}) were {percentage_diff:.1f}% higher "
            f"than sales during non-holiday weeks (${non_holiday_sales:,.0f}).\n"
            f"Chart saved to: {chart_path}"
        )
        return summary

    def top_performing_stores(self, top_n: int = 5) -> str:
        """
        Analyzes the top performing stores in the given data.
        :param top_n: number of top performers to analyze
        :return: textual summary of top performers
        """
        attrUse, colUse = sales_attribute(isDaily=False)
        targetDf: pd.DataFrame = self.gen_df

        print_stderr("generating top performers bar graph")
        sales_by_store = targetDf.groupby('Store', as_index=False)[colUse].sum()
        sales_by_store = sales_by_store.sort_values(by=colUse, ascending=False)

        chart_path = self.output_dir / "top_performers.png"
        if not os.path.exists(chart_path):
            title = f"Top {top_n} Performing Stores by Total Sales"
            fig = px.bar(
                sales_by_store.head(top_n),
                x='Store',
                y=colUse,
                title=title,
                labels={'Store': 'Store ID', colUse: 'Total Sales ($)'},
                text_auto='.2s'
            )
            fig.write_image(chart_path)

        summary = (
            f"Top performers:\n"
        )
        for i in range(min(top_n, len(sales_by_store))):
            summary += f"Store {sales_by_store.iloc[i]['Store']:,.0f}: {sales_by_store.iloc[i][colUse]:,.0f}\n"
        summary += f"Chart saved to: {chart_path}"
        return summary

    def department_analysis_no_holiday(self, top_n: int = 5) -> str:
        """
        Analyzes top performing departments over all time in given data.
        :param top_n: number of top performing departments to analyze
        :return: textual summary of departments
        """
        attrUse, colUse = sales_attribute(isDaily=False)
        targetDf: pd.DataFrame = self.gen_df

        print(f"\n\n\n{targetDf.head()}\n\n\n")
        targetDf.loc[:, 'Dept'] = targetDf['Dept'].apply(get_department_name)
        sales_by_dept = targetDf.groupby('Dept', as_index=False)[colUse].sum()
        sales_by_dept = sales_by_dept.sort_values(by=colUse, ascending=False)

        chart_path = self.output_dir / "departments.png"
        if not os.path.exists(chart_path):
            fig = px.bar(
                sales_by_dept.head(top_n),
                x='Dept',
                y=colUse,
                title=f"Top {top_n} Performing Depts by Total Sales (All Time)",
                labels={'Dept': 'Dept ID', colUse: 'Total Sales ($)'},
                text_auto='.2s'
            )
            fig.write_image(chart_path)

        top_dept = sales_by_dept.iloc[0]
        bottom_dept = sales_by_dept.iloc[-1]

        summary = (
            f"The following department data is over general course of all time. Dept {top_dept['Dept']} was the top performer with total sales of ${top_dept[colUse]:,.0f}. "
            f"In contrast, Dept {bottom_dept['Dept']} had the lowest sales with a total of ${bottom_dept[colUse]:,.0f}.\n"
        )
        summary += f"Chart saved to: {chart_path}"
        return summary

    def department_analysis_holiday(self, top_n: int = 5) -> str:
        """
        Analyzes top performing departments over only holidays in given data.
        :param top_n: number of top performing departments to analyze
        :return: textual summary of top departments
        """
        attrUse, colUse = sales_attribute(isDaily=False)
        targetDf: pd.DataFrame = self.gen_df

        print(f"\n\n\nDept Analysis Holiday:\n{targetDf.head()}\n\n\n")
        targetDf.loc[:, 'Dept'] = targetDf['Dept'].apply(get_department_name)
        targetDf = targetDf[targetDf['IsHoliday'] == True]
        sales_by_dept = targetDf.groupby('Dept', as_index=False)[colUse].sum()
        sales_by_dept = sales_by_dept.sort_values(by=colUse, ascending=False)

        chart_path = self.output_dir / "departments_hol.png"
        if not os.path.exists(chart_path):
            fig = px.bar(
                sales_by_dept.head(top_n),
                x='Dept',
                y=colUse,
                title=f"Top {top_n} Performing Depts by Total Sales (Only Holidays)",
                labels={'Dept': 'Dept ID', colUse: 'Total Sales ($)'},
                text_auto='.2s'
            )
            fig.write_image(chart_path)

        top_dept = sales_by_dept.iloc[0]
        bottom_dept = sales_by_dept.iloc[-1]

        summary = (
            f"The following department data is over only holidays. Dept {top_dept['Dept']} was the top performer with total sales of ${top_dept[colUse]:,.0f}. "
            f"In contrast, Dept {bottom_dept['Dept']} had the lowest sales with a total of ${bottom_dept[colUse]:,.0f}.\n"
        )
        summary += f"Chart saved to: {chart_path}"
        return summary

    def analyze_economic_headwinds(self) -> str:
        """
        Creating a propensity score for measuring external factors against performance. Here, forecasting is avoided since
        it'll be hard to predict external factors of the market with only these paremeters. However, this may serve towards
        1. Comparing with weekly sales to examine responses to external factors
        2. Is the budget flexible enough to react to changes?
        :return: textual summary of the most recent trend, 12 week trends, Y-o-Y trends, overall avg, max, min and volatility.
        """

        # ideal temperature in Celsius. We measure the distance from this.
        # Since we want all factors to be proportional for growth, we invert this score.

        spec_df = self.spec_df
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

        chart_path = self.output_dir / "propensity.png"
        if not os.path.exists(chart_path):
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(14, 7))

            ax.plot(spec_df.index, spec_df['headwinds_index'], label='Retail Headwinds Index', color='b', lw=2)
            ax.set_title('Weekly Retail Headwinds Index (2-Year Period)', fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Index Score (Higher = More Favorable)', fontsize=12)
            ax.legend()
            ax.grid(True)
            plt.savefig(chart_path)

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
        summary.append(f"\nChart saved to: {chart_path}")

        return " ".join(summary)

    def analyze_event_impact(self, event_query: str) -> str:
        """
        Analyzes the sales impact of a specific business event, like a marketing campaign or product launch.
        Use this to understand why sales may have changed during a specific period by correlating it with a known event.
        :param event_query: event name to analyze
        """

        event = self.log.find_event(event_query)
        if not event:
            return f"I could not find any event in my log related to '{event_query}'."

        # Define the event period and a baseline period for comparison
        event_period_df = self.gen_df[
            (self.gen_df['Date'] >= event.start_date) & (self.gen_df['Date'] <= event.end_date)
            ]

        baseline_start = event.start_date - pd.Timedelta(weeks=4)
        baseline_end = event.start_date - pd.Timedelta(days=1)
        baseline_df = self.gen_df[
            (self.gen_df['Date'] >= baseline_start) & (self.gen_df['Date'] <= baseline_end)
            ]

        if event_period_df.empty:
            return f"Found event '{event.description}', but no sales data exists for its date range."

        avg_event_sales = event_period_df['Weekly_Sales'].mean()
        avg_baseline_sales = baseline_df['Weekly_Sales'].mean() if not baseline_df.empty else 0

        if avg_baseline_sales == 0:
            return (f"During the event '{event.description}', average weekly sales were ${avg_event_sales:,.2f}. "
                    "No baseline data is available for comparison.")

        percentage_change = ((avg_event_sales - avg_baseline_sales) / avg_baseline_sales) * 100

        return (f"Analysis for event '{event.description}' (from {event.start_date.date()} to {event.end_date.date()}):\n"
                f"- Average weekly sales during the event were ${avg_event_sales:,.2f}.\n"
                f"- This represents a {percentage_change:+.1f}% change compared to the 4-week baseline average of ${avg_baseline_sales:,.2f}.")

    # def _generate_predictions(self) -> pd.DataFrame:
    #     targetDf = self.gen_df
    #     targetDf.loc[:, 'Dept'] = targetDf['Dept'].apply(get_department_id)
    #     train_data = TimeSeriesDataFrame.from_data_frame(
    #         targetDf,
    #         id_column="Store",
    #         timestamp_column="Date"
    #     )
    #
    #     predictor = self.predictor
    #     predictions = predictor.predict(train_data)
    #     chart_path = self.output_dir / "forecasted.png"
    #     if not os.path.exists(chart_path):
    #         fig = predictor.plot(
    #             train_data,
    #             predictions,
    #             max_history_length=150,
    #             item_ids=[1]
    #         )
    #         fig.savefig(chart_path)
    #     pred_df = predictions.reset_index()
    #     pred_df.rename(columns={"item_id": "Store", "timestamp": "Date", "mean": "Weekly_Sales"}, inplace=True)
    #     pred_df["Weekly_Sales"] = pred_df["Weekly_Sales"].apply(lambda x: round(x, 2))
    #
    #     self.forecast_plot_path = chart_path
    #     self.forecast_df = pred_df
    #
    #     return pred_df
    #
    # def forecast_weekly_sales(self, num_weeks: int = 12) -> str:
    #     """
    #     Generates a sales forecast for a specified number of future weeks.
    #     Runs a predictive model and returns a summary of the forecast.
    #     """
    #     pred_df = self._generate_predictions()
    #     self.pred_df = pred_df[['Store', 'Date', 'Weekly_Sales']].copy()
    #     forecast_period = pred_df.head(num_weeks)
    #     avg_predicted_sales = forecast_period['Weekly_Sales'].mean()
    #     peak_sales_date = forecast_period.loc[forecast_period['Weekly_Sales'].idxmax()]['Date'].date()
    #     peak_sales_value = forecast_period['Weekly_Sales'].max()
    #     chart_path = self.output_dir / "forecasted.png"
    #     summary = (
    #         f"Sales forecast for the next {num_weeks} weeks generated successfully.\n"
    #         f"- The average predicted weekly sales are ${avg_predicted_sales:,.2f}.\n"
    #         f"- The forecast peaks at ${peak_sales_value:,.2f} on {peak_sales_date}.\n"
    #         f"- Chart saved to {chart_path}"
    #     )
    #     return summary
    def _generate_predictions(self, num_weeks: int = 12) -> Tuple[pd.DataFrame, str]:
        targetDf = self.gen_df
        targetDf.loc[:, 'Dept'] = targetDf['Dept'].apply(get_department_id)
        forecast_df, summary = hierarchical_forecast_with_reconciliation(gen_df=targetDf, forecast_periods=num_weeks)
        pred_df = forecast_df.reset_index()
        pred_df.rename(columns={"item_id": "Store", "timestamp": "Date", "mean": "Weekly_Sales"}, inplace=True)
        pred_df["Weekly_Sales"] = pred_df["Weekly_Sales"].apply(lambda x: round(x, 2))

        self.forecast_plot_path = self.output_dir / "forecasted.png"
        self.forecast_df = pred_df

        return pred_df, summary

    def forecast_weekly_sales(self, num_weeks: int = 12) -> str:
        """
        Generates a sales forecast for a specified number of future weeks.
        Runs a predictive model and returns a summary of the forecast.
        """
        pred_df, summary = self._generate_predictions(num_weeks=num_weeks)
        self.pred_df = pred_df[['Date', 'Store', 'Dept', 'Weekly_Sales']].copy()
        store_is_numeric = pd.to_numeric(self.pred_df['Store'], errors='coerce').notna()
        dept_is_numeric = pd.to_numeric(self.pred_df['Dept'], errors='coerce').notna()
        self.pred_df = self.pred_df[store_is_numeric & dept_is_numeric].copy()
        return summary
