import random
import datetime
import holidays
import pandas as pd

from src.utils import stderr_print


def generate_months(n_months: int = 3):
    data = [
        ['Store', 'Dept', 'Date', 'Daily_Sales', 'IsHoliday']
    ]
    mu, sigma = 20, 100
    us_holidays = holidays.US()
    current_date = datetime.datetime(2013, 1, 1)
    for day in range(91): # Jan - Mar 2013
        isHoliday: bool = True if current_date in us_holidays else False
        for store in range(1, 46):
            for department in range(1, 99):
                if (department % 11 == 0): continue # some randomization and skipping
                today_sales_d_s = max(0, (random.normalvariate(mu, sigma) * 0.1)) # today's sales for this dept in this store
                formatted_date = f"{current_date.month}/{current_date.day}/{current_date.year % 100}"
                data.append([store, department, formatted_date, today_sales_d_s, isHoliday])
        current_date = current_date + datetime.timedelta(days=1)
    dfdata = pd.DataFrame.from_records(data)
    stderr_print(dfdata)
    dfdata.to_csv('test_data/train_daily.csv', index=False, header=False)


def check():
    generate_months()