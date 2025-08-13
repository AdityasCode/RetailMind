import os
from autogluon.timeseries import TimeSeriesPredictor
from src.crud import Event, EventLog, CRUD
from src.eda import EDAFeatures
from src.model_training import hierarchical_forecast_with_reconciliation
from src.testing.generate_daily_csv import check
from src.utils import print_stderr, default_storeIDs, parse_pdf

# ideally should be an ipynb but directory mapping is a significantly bigger issue

def cwd_test():
    print_stderr(f"cwd: {os.getcwd()}")
# from generate_daily_csv import check
# check()
# from src.crud import WeekCRUD
# w = WeekCRUD()

def logTest():
    myEvent = Event(
        text="""
        Our Q3 marketing push, the Summer Sizzler campaign, focused on boosting sales of outdoor and grocery products through targeted digital ads and in-store promotions.
        """,
        name="Summer Sizzler Marketing Campaign",
        start_date="2010-07-02",
        end_date="2010-07-23",
    )
    myLog = EventLog()
    myLog.add_event_from_event(myEvent)
    print_stderr(myLog)
    print_stderr(myLog.log)
    print_stderr(myLog.toString())
    return myLog

def dailyCRUDTest(crudder):
    print_stderr(crudder.gen_df)
    print_stderr(crudder.daily_df)

def agentTest(retail_chatbot):
    question1 = "What were the total sales for store 2 in 2011?"
    answer1 = retail_chatbot.ask(question1)
    print(f"\nAnswer: {answer1}")
    question2 = "Show me an analysis of the top 3 performing stores."
    answer2 = retail_chatbot.ask(question2)
    print(f"\nAnswer: {answer2}")
    question3 = "List the total sales for those stores in 2011."
    answer3 = retail_chatbot.ask(question3)
    print(f"\nAnswer: {answer3}")
    question4 = "Tell me everything you can about this data, including stores, holiday performance, top performers, retail headwinds, etc."
    answer4 = retail_chatbot.ask(question4)
    print(f"\nAnswer: {answer4}")
    question5 = "Discuss the impact of the sales campaign event on sales."
    answer5 = retail_chatbot.ask(question5)
    print(f"\nAnswer: {answer5}")

def headwinds_test(edaer: EDAFeatures):
    print_stderr(edaer.analyze_economic_headwinds())

def top_test(edaer: EDAFeatures):
    print(edaer.top_performing_stores())

def pred_test(edaer: EDAFeatures):
    x = edaer.forecast_weekly_sales()
    print_stderr("pred test")
    print_stderr(x)
    print_stderr("pred_df:")
    print_stderr(edaer.pred_df)

def run_hierarchical_forecast_test(edaer: EDAFeatures):
    """
    Example of how to use the hierarchical forecasting function
    """
    summary = edaer.forecast_weekly_sales(
        num_weeks=24,  # 6 months ahead
    )

    print_stderr(summary)
    return summary

def sales_test(edaer: EDAFeatures):
    print_stderr(edaer.sales_t())

def pdf_test():
    print_stderr(parse_pdf("../test_data/InvChanges.pdf"))

def main():
    cwd_test()
    # check()
    crudder = CRUD()
    # log = logTest()
    # edaer = EDAFeatures(crudder.gen_df, crudder.spec_df, daily_df=crudder.daily_df, event_log=log,
    # storeIDs=default_storeIDs, predictor=TimeSeriesPredictor.load("../models/autogluon-m4-hourly"))
    # edaer = EDAFeatures(crudder.gen_df, crudder.spec_df, daily_df=crudder.daily_df,
    #                     event_log=None, storeIDs=default_storeIDs, predictor=None)
    # retail_chatbot = RetailAgent(crudder, edaer)
    # logTest()
    # dailyCRUDTest(crudder)
    # agentTest(retail_chatbot)
    # headwinds_test(edaer)
    # top_test(edaer)
    # pred_test(edaer=edaer)
    # run_hierarchical_forecast_test(edaer)
    # sales_test(edaer)
    # pdf_test()


if __name__ == "__main__":
    main()