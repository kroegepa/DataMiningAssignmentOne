import pandas as pd
from matplotlib import pyplot as plt
import pandas as pds
import numpy as np
from datetime import date, datetime, time
from tqdm import tqdm
from pandas import Timestamp

path = "datasets/dataset_mood_smartphone.csv"

dataset_df = pd.read_csv(path, index_col=0)
dataset_df["time"] = pd.to_datetime(dataset_df['time'])

# dataset_df.info()
# print(dataset_df.head())

var_names = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', "screen",
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
    '''
    Gives all the rows that fit the params. (for indiv between time1 and time2)
    :param indiv: individual id
    :param date_time1: first date time
    :param date_time2: second date time
    :param df: dataframe of the dataset
    :return: DataFrame object
    '''
    if date_time1 > date_time2:
        date_time1, date_time2 = date_time2, date_time1

    result = []

    for index, row in df[df['id'] == indiv].iterrows():
        if date_time1 <= row["time"] < date_time2:
            result.append(row)

    return pd.DataFrame(result, columns=list(df.columns))


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


def create_per_interval_and_participant_dataset(df, save_path="per_day_participant_dataset.csv", interval=24):
    """
    Aggregates data given the number of hours as interval
    """
    if df["time"].dtype != 'datetime64[ns]':
        df["time"] = pd.to_datetime(df['time'], format='mixed')
        # df["time"] = pd.to_datetime(df['time'])

    # datapoint for each day + individual combination (good?)
    unique_ids = get_unique_individual_ids(df)
    datapoints = {}
    for participant_id in tqdm(unique_ids, desc="participants"):
        # dataframe containing only rows of participant with id "participant_id"
        id_df = df[df["id"] == participant_id]

        min_time = id_df['time'].min()
        max_time = id_df['time'].max()

        # convert min and max time to date nearest date
        min_time = datetime.combine(min_time.date(), time(0, 0))  # floor to nearest date
        max_time = max_time.date() + pd.Timedelta(days=1)  # ceil to nearest date

        time_range = pd.to_datetime(pd.date_range(min_time, max_time, freq=f"{interval}h"))

        for time_i, start_time in enumerate(time_range):
            if time_i == len(time_range) - 1:
                break
            if time_range[time_i + 1] - time_range[time_i] != pd.Timedelta(f"{interval}h"):
                break

            end_time = time_range[time_i + 1]

            relevant_rows = get_all_rows_between(participant_id, start_time, end_time, id_df)
            datapoints.setdefault((participant_id, start_time), [0] * len(var_and_count_names))
            for index, row in relevant_rows.iterrows():
                var_index = var_and_count_names.index(row["variable"])
                datapoints[(participant_id, start_time)][var_index] += row["value"]
                datapoints[(participant_id, start_time)][var_index + 1] += 1

    datapoints_list = []
    # transform dict to list
    for key, value in datapoints.items():
        temp = [key[0], key[1]]
        temp += value
        datapoints_list.append(temp)

    # create dataframe from dict and save it to save_path
    datapoints_df = pd.DataFrame(datapoints_list, columns=["id", "date"] + var_and_count_names)
    datapoints_df.info()
    datapoints_df.to_csv(save_path, index=False)


def drop_counts_from_df(load_path="per_day_participant_dataset.csv", save_path="sum_dataset.csv"):
    """
    drops count vars from the df
    :param load_path: loads path
    :param save_path: save path
    :return: None
    """
    df = pd.read_csv(load_path)
    df = df.drop(count_var_names, axis=1)
    df.to_csv(save_path, index=False)


def count_dataset_to_avg_dataset(load_path="per_day_participant_dataset.csv", save_path="avg_dataset.csv",
                                 drop_counts=False, avg_names=None):
    """
    loads the dataset and by default averages the first 4 vars using the counts
    :param drop_counts: True = drops counts
    :param load_path: path it loads
    :param save_path: path it saves to
    :param avg_names: default value: ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    :return: None
    """
    df = pd.read_csv(load_path)
    if avg_names is None:
        # avg_names = var_names
        avg_names = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']

    for avg_name in avg_names:
        df[avg_name] = np.where(df[avg_name + "_count"] == 0, 0, df[avg_name] / df[avg_name + "_count"])
    if drop_counts:
        df = df.drop(count_var_names, axis=1)
    df.to_csv(save_path, index=False)


def transform_data(load_path="dataset_mood_smartphone.csv", save_path="dataset_mood_smartphone_with_co.csv",
                   cutoff=cutoff_list):
    df = pd.read_csv(load_path)
    if df["time"].dtype != 'datetime64[ns]':
        df["time"] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f')
    for index, name in enumerate(var_names):
        if index > 2:
            mean_count = 0
            mean = 0
            if cutoff[index] != None:

                # Compute mean for values within the cutoff range
                for row_index, row in df[df['variable'] == name].iterrows():
                    if row["value"] < 0:
                        pass
                    elif row["value"] > cutoff[index]:
                        pass
                    else:
                        mean_count += 1
                        mean += row["value"]

                # Set negative values to 0
                df.loc[(df['value'] < 0) & (df['variable'] == name), "value"] = 0

                mean = mean / mean_count

                # Set values above cutoff to computed mean
                df.loc[(df['value'] > cutoff[index]) & (
                        df['variable'] == name), "value"] = mean  # should't we replace it with cutoff[index]?

    df.to_csv(save_path, index=False)


def reformat_aggregated_data(df):
    df["date_time"] = pd.to_datetime([eval(string)[1] for string in df['Unnamed: 0']])
    df["participant_id"] = [eval(string)[0] for string in df['Unnamed: 0']]

    df.drop(columns=['Unnamed: 0'], inplace=True)
    df = df[df.columns.to_list()[-2:] + df.columns.to_list()[:-2]]

    return df


# def avarage_data(df, avarage_vars = var_names[0:4]):
#     for var in avarage_vars:
#         df[var] = df[var]/df[var + "_count"]
#     return df

# set to zero if division by zero
def average_data(df, avarage_vars=None):
    if avarage_vars is None:
        avarage_vars = var_names[0:4]

    for var in avarage_vars:
        df[var] = np.where(df[var + "_count"] == 0, 0, df[var] / df[var + "_count"])
    return df


def plot_counts_per_participant(df):
    participants = df['participant_id'].unique()
    for participant in participants:
        plt.plot(df[df['participant_id'] == participant]['date_time'],
                 df[df['participant_id'] == participant]['mood_count'])
        plt.title('Participant ' + participant + ' mood count')
        plt.show()
        plt.plot(df[df['participant_id'] == participant]['date_time'],
                 df[df['participant_id'] == participant].iloc[:, 3::2].sum(axis=1))
        plt.title('Participant ' + participant + ' sum of the _count variables')
        plt.show()


# transform_data()
if __name__ == "__main__":
    # transform the data
    transform_data("datasets/dataset_mood_smartphone_no_nan.csv", "datasets/dataset_mood_smartphone_co.csv")
    # load transformed dataset
    transformed_df = pd.read_csv("datasets/dataset_mood_smartphone_co.csv")
    # create per day and participant dataset
    create_per_interval_and_participant_dataset(transformed_df, "datasets/per_day_participant_dataset_co.csv", interval=24)
    # calc avg of the first 4 vars
    count_dataset_to_avg_dataset(load_path="datasets/per_day_participant_dataset_co.csv",
                                 save_path="datasets/per_day_participant_dataset_co_w_avg.csv")