import pandas as pd
from matplotlib import pyplot as plt
import pandas as pds
import numpy as np
from datetime import date, datetime, time

dataset_df = pd.read_csv("dataset_mood_smartphone.csv", index_col=0)
dataset_df["time"] = pd.to_datetime(dataset_df['time'])
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


# testing get_all_rows_between()
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


def get_unique_dates(df):
    """
    returns unique dates in the df
    :param df: dataframe
    :return: list containing datetime.date type objects
    """
    new_df = pd.to_datetime(df['time']).dt.date
    return new_df.unique()


def get_unique_individual_ids(df):
    """
    returns unique individual_ids in the df
    :param df: dataframe
    :return: list containing individual_ids as string
    """
    return df["id"].unique()


unique_ids = get_unique_individual_ids(dataset_df)
for participant_id in unique_ids:
    # dataframe containing only rows of participant with id "participant_id"
    id_df = dataset_df[dataset_df["id"] == participant_id]
    unique_dates = get_unique_dates(id_df)
    for date in unique_dates:
        # per date
        time1 = datetime.combine(date, time(0, 0))
        time2 = datetime.combine(date, time(23, 59))
        relevant_rows = get_all_rows_between(participant_id, time1, time2, id_df)
