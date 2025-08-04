import random
import datetime
from typing import List

import holidays
import pandas as pd

from src.utils import print_stderr


def generate_months(n_months: int = 3):
    data: pd.DataFrame | List = [
        ['Store', 'Dept', 'Date', 'Daily_Sales', 'IsHoliday']
    ]
    mu, sigma = 20, 100
    us_holidays = holidays.US()
    current_date = datetime.datetime(2013, 1, 1)
    for day in range(1, 15): # Jan 1 - Jan 14; 2 weeks
        isHoliday: bool = True if current_date in us_holidays else False
        for store in range(1, 46):
            for department in range(1, 99):
                if (department % 11 == 0): continue # some randomization and skipping
                today_sales_d_s = round(max(0, (random.normalvariate(mu, sigma) * 0.1)), 2) # today's sales for this dept in this store
                formatted_date = f"{current_date.month}/{current_date.day}/{current_date.year % 100}"
                data.append([store, department, formatted_date, today_sales_d_s, isHoliday])
        if day % 7 == 0:
            data = pd.DataFrame.from_records(data)
            print_stderr(data)
            data.to_csv(f'test_data/train_daily_{2 if day % 14 == 0 else 1}.csv', index=False, header=False)
            data = []
        current_date = current_date + datetime.timedelta(days=1)



def check():
    generate_months()