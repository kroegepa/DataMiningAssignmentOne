import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt

#Calculate and show correlation matrix
def calc_corr_matrix():
    path = "sum_dataset.csv"
    #path = "dataset_mood_smartphone.csv"

    df = pd.read_csv(path, index_col=0)
    df.info()
    #print(df.head())

    # for sum dataset
    var_names = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen',
                'call', 'sms', 'appCat.builtin', 'appCat.communication',
                'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
                'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
                'appCat.utilities', 'appCat.weather']
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df.info()
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.show()

def calc_var_properties():
    path = "sum_dataset.csv"
    #path = "dataset_mood_smartphone.csv"

    analysis_df = pd.read_csv(path, index_col=0)
    analysis_df.info()
    print(analysis_df.head())

    # for sum dataset
    var_names = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen',
                'call', 'sms', 'appCat.builtin', 'appCat.communication',
                'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
                'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
                'appCat.utilities', 'appCat.weather']

    # count
    count_dict = analysis_df.groupby("variable")["value"].count().to_dict()
    # print(analysis_df.groupby("variable")["value"].count())
    # mean
    mean_dict = analysis_df.groupby("variable")["value"].mean().to_dict()
    # median
    median_dict = analysis_df.groupby("variable")["value"].median().to_dict()
    # max
    max_dict = analysis_df.groupby("variable")["value"].max().to_dict()
    # min
    min_dict = analysis_df.groupby("variable")["value"].min().to_dict()
    # std
    std_dict = analysis_df.groupby("variable")["value"].std().to_dict()
    # how many unique dates?
    temp_df = analysis_df
    temp_df["time"] = pd.to_datetime(temp_df['time']).dt.date
    unique_dates_dict = analysis_df.groupby("variable")["time"].nunique().to_dict()
    # is value missing ?
    temp_df = analysis_df
    temp_df['value'] = analysis_df.isna()['value'].values
    nan_dict = temp_df.groupby("variable")["value"].value_counts().to_dict()

    stats = {
        "count": count_dict,
        "mean": mean_dict,
        "median": median_dict,
        "max": max_dict,
        "min": min_dict,
        "std": std_dict,
        "unique_dates": unique_dates_dict,
        "nan_count": nan_dict
    }

    tablulate_list = []
    for var_name in var_names:
        # print(tabulate([['Alice', 24], ['Bob', 19]], headers=['Name', 'Age']))
        row = [var_name]
        for stat in stats.keys():
            if stat != "nan_count":
                row.append(stats[stat][var_name])
            else:
                if (var_name, True) in stats[stat].keys():
                    row.append(stats[stat][(var_name, True)])
                else:
                    row.append(0)
        tablulate_list.append(row)

    print(tabulate(tablulate_list, headers=["Variable"] + list(stats.keys()), tablefmt='outline'))

