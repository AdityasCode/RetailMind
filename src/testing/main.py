import os
print(f"cwd: {os.getcwd()}")
from generate_daily_csv import check
check()
from src.crud import WeekCRUD
w = WeekCRUD()