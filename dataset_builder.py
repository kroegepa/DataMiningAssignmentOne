import pandas as pd
import datetime as dt

# take one day
# look if there is a day + 1 * k times + 1 for label
# if count(missing days) below threshold don't add
#

path = "per_day_participant_dataset.csv"
dataset_df = pd.read_csv(path)
dataset_df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

# k = how big should the "history" be?, how many days should be in a datapoint
k = 5
missing_days_threshold = 1


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

    for index, row in df[df['index'] == indiv].iterrows():
        if date_time1 >= row["date"] >= row["date"]:
            result.append(row)

    return pd.DataFrame(result, columns=list(df.columns))


# read one row
# does label exist? (k+1)
# get all rows between its day and it + k days
for index, row in dataset_df.iterrows():
    # transforms string index to working tuple containing (participant(str),date (datetime.date))
    participant_id, current_date = eval(row[0], {"__builtins__": {}}, {'datetime': dt})

    # check if label exists:
    if str((participant_id, current_date + dt.timedelta(days=k))) in dataset_df["index"].values:
        rows = get_all_rows_between(participant_id, current_date, current_date + dt.timedelta(days=k - 1), dataset_df)
        # only continue if there are under missing_days_threshold missing days
        if rows.shape[0] > k - missing_days_threshold:
            pass

    break
