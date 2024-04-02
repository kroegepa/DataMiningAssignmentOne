import pandas as pd
from matplotlib import pyplot as plt
import pandas as pds
import numpy as np

dataset_df = pd.read_csv("dataset_mood_smartphone.csv", index_col=0)
dataset_df.info()
print(dataset_df.head())


def get_all_rows_between(indiv, date_time1, date_time2, df=dataset_df):
    '''
    Gives all the rows that fit the params. (for indiv between time1 and time2)
    :param indiv: individual id
    :param date_time1: first date time
    :param date_time2: second date time
    :param df: dataframe of the dataset
    :return: DataFrame object
    '''
    if date_time1 < date_time2:
        date_time1, date_time2 = date_time2, date_time1

    result = []

    for index, row in df[df['id'] == indiv].iterrows():
        if date_time1 >= row["time"] >= row["time"]:
            result.append(row)

    return pd.DataFrame(result, columns=list(df.columns))


# testing
'''
print(f"\n{dataset_df.iloc[0]}")
time1 = dataset_df.iloc[0]["time"]
time2 = dataset_df.iloc[1]["time"]
indiv_id = dataset_df.iloc[0]["id"]
print(f"{time1} to {time2}")

resulting_df = get_all_rows_between(indiv_id, time1, time2)
resulting_df.info()
print(resulting_df.head())
'''
