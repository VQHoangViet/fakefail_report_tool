from imports import *

if __name__ == '__main__':
    start_day = input('Start day: ')
    end_day = input('End day: ')
    if end_day == '':
        end_day = str(pd.to_datetime(start_day) - relativedelta(days=2))
    url = input('Link POD manual check: ')
    ff.pipeline(start_day, end_day, url)