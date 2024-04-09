import pandas as pd
from matplotlib import pyplot as plt
import pandas as pds
import numpy as np
from datetime import date, datetime, time
from tqdm import tqdm

# path = "small_adv_dataset.csv"
path = "dataset_mood_smartphone.csv"

dataset_df = pd.read_csv(path, index_col=0)
dataset_df["time"] = pd.to_datetime(dataset_df['time'])

# dataset_df.info()
# print(dataset_df.head())

var_names = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen',
             'call', 'sms', 'appCat.builtin', 'appCat.communication',
             'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
             'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
             'appCat.utilities', 'appCat.weather']

cutoff_list = [None, None, None, None, 2000, None, None, 1000, 2000, 5000, None, None, 5000, None, 5000, 2000, None,
               None, None]
cutoff_list_test = [None, None, None, None, 2000, None, None, 1000, 2000, 5000, None, 2000, 5000, 1000, 5000, 2000,
                    1500, 500, None]
count_var_names = ["mood_count", "circumplex.arousal_count", "circumplex.valence_count", "activity_count",
                   "screen_count", "call_count", "sms_count", "appCat.builtin_count", "appCat.communication_count",
                   "appCat.entertainment_count", "appCat.finance_count", "appCat.game_count", "appCat.office_count",
                   "appCat.other_count", "appCat.social_count", "appCat.travel_count", "appCat.unknown_count",
                   "appCat.utilities_count", "appCat.weather_count"]

var_and_count_names = ["mood", "mood_count", "circumplex.arousal", "circumplex.arousal_count", "circumplex.valence",
                       "circumplex.valence_count", "activity", "activity_count", "screen", "screen_count", "call",
                       "call_count",
                       "sms", "sms_count", "appCat.builtin", "appCat.builtin_count", "appCat.communication",
                       "appCat.communication_count", "appCat.entertainment", "appCat.entertainment_count",
                       "appCat.finance",
                       "appCat.finance_count", "appCat.game", "appCat.game_count", "appCat.office",
                       "appCat.office_count",
                       "appCat.other", "appCat.other_count", "appCat.social", "appCat.social_count", "appCat.travel",
                       "appCat.travel_count", "appCat.unknown", "appCat.unknown_count", "appCat.utilities",
                       "appCat.utilities_count", "appCat.weather", "appCat.weather_count"]


def get_all_rows_between(indiv, date_time1, date_time2, df=dataset_df):
    """
    Gives all the rows that fit the params. (for indiv between time1 and time2)
    :param indiv: individual id
    :param date_time1: first date time
    :param date_time2: second date time
    :param df: dataframe of the dataset
    :return: DataFrame object
    """
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


def get_unique_variables(df):
    """
    returns unique individual_ids in the df
    :param df: dataframe
    :return: list containing individual_ids as string
    """
    return df["variable"].unique()


def create_per_day_and_participant_dataset(save_path="per_day_participant_dataset.csv"):
    # var_names = get_unique_variables(dataset_df) # don't use this, it's a numpy array
    """
    var_names = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen',
                 'call', 'sms', 'appCat.builtin', 'appCat.communication',
                 'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
                 'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
                 'appCat.utilities', 'appCat.weather']
    """

    # datapoint for each day + individual combination (good?)
    unique_ids = get_unique_individual_ids(dataset_df)
    datapoints = {}
    for participant_id in tqdm(unique_ids, desc="participants"):
        # dataframe containing only rows of participant with id "participant_id"
        id_df = dataset_df[dataset_df["id"] == participant_id]
        unique_dates = get_unique_dates(id_df)
        for date in unique_dates:
            # per date
            time1 = datetime.combine(date, time(0, 0))
            time2 = datetime.combine(date, time(23, 59))
            relevant_rows = get_all_rows_between(participant_id, time1, time2, id_df)
            datapoints.setdefault((participant_id, date), [0] * len(var_and_count_names))
            for index, row in relevant_rows.iterrows():
                var_index = var_and_count_names.index(row["variable"])
                datapoints[(participant_id, date)][var_index] += row["value"]
                datapoints[(participant_id, date)][var_index + 1] += 1

    datapoints_list = []
    # transform dict to list
    for key, value in datapoints.iteritems():
        temp = [key[0], key[1]]
        temp += value
        datapoints_list.append(temp)

    # create dataframe from dict and save it to save_path
    datapoints_df = pd.DataFrame.from_dict(datapoints, orient='index', columns=["id", "date"] + var_and_count_names)
    datapoints_df.info()
    datapoints_df.to_csv(save_path)


def pdp_dataset_to_sum_dataset(load_path="per_day_participant_dataset.csv", save_path="sum_dataset.csv"):
    df = pd.read_csv(load_path)
    df = df.drop(count_var_names, axis=1)
    df.to_csv(save_path)


def pdp_dataset_to_avg_dataset(load_path="per_day_participant_dataset.csv", save_path="avg_dataset.csv"):
    df = pd.read_csv(load_path)

    for var_name in var_names:
        df[var_name] = df[var_name] / df[var_name + "_count"]

    df = df.drop(count_var_names, axis=1)
    df.to_csv(save_path)


def transform_data(load_path="per_day_participant_dataset.csv", save_path="per_day_participant_dataset_with_co.csv",
                   cutoff=cutoff_list):
    df = pd.read_csv(load_path, index_col=0)
    df["time"] = pd.to_datetime(dataset_df['time'])
    for index, name in enumerate(var_names):
        if index > 2:
            df.loc[df[name] < 0, name] = 0
            mean = df.loc[df[name] < cutoff[index], name].mean()
            df.loc[df[name] > cutoff[index], name] = mean
    df.to_csv(save_path)


transform_data()
# create_per_day_and_participant_dataset()
# pdp_dataset_to_sum_dataset()
# pdp_dataset_to_avg_dataset()
