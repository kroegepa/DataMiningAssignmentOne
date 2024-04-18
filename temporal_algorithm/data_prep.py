import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# parameters
show_histo = False
show_output = False
show_extended_output = False
normalize = True

seq_length = 5

# read dataset
df = pd.read_csv("../datasets/filled_complete_dataset.csv")

# data analysis
if show_histo:
    plt.figure()
    df["mood"].hist(alpha=0.75, bins=20)
    plt.title(f"Histogram of values for mood")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.xticks(ticks=range(11))
    plt.grid(False)
    plt.show()

# create equal size bins + add classes to rows
df["mood_class"], quantile_bins = pd.qcut(df["mood"], q=3, labels=[0, 1, 2], retbins=True)
if show_output:
    print(df["mood_class"].describe())
    print(df["mood"].tail())
    print(df["mood_class"].tail())
    print(quantile_bins)

    #############################
    # creating temporal dataset #
    #############################

# drop rows where dates are not in sequence
indices_to_drop = df[df["batch"] == 1].index
df = df.drop(indices_to_drop)

# check if dates are in sequence
df["date_temp"] = pd.to_datetime(df["date"])
for name, group_id in df.groupby("id", sort=False):
    # Calculate difference and check for consecutive days
    group_id["diff"] = group_id["date_temp"].diff(periods=-1).abs()
    # Calculate absolute difference to handle negative values
    are_consecutive = (group_id["diff"][:-1] == pd.Timedelta(days=1)).all()
    if show_output:
        print(f"Are all dates consecutive for {name}? {are_consecutive}")
    # if output_prints:
    if not are_consecutive:
        print(f"Are all dates consecutive for {name}? {are_consecutive}")
        print(group_id["diff"])


# normalize data

# Initialize the MinMaxScaler
scaler = MinMaxScaler()
# Fit and transform the data
#df['A_min_max_scaled'] = scaler.fit_transform(df[['A']])

# drop useless columns
df = df.drop(columns=["date_temp", "batch"])
if show_output:
    print(df.columns)

# create sequences
seq_X = []
seq_Y = []
for name, group_id in df.groupby("id", sort=False):
    # check if there are enough rows for sequence
    if len(group_id) > seq_length:
        # for each participant take the first seq_length rows then do it +1
        for index in range(len(group_id) - seq_length - 1):
            seq_X.append(group_id.iloc[index:index + seq_length].drop(columns=["date", "id"]).to_numpy())
            seq_Y.append(group_id.iloc[index + seq_length + 1].drop(columns=["date", "id"]).to_numpy()[-1])

if show_output:
    print(seq_X)
    print(seq_Y)

np.savez("dataset_not_normalized.npz", array1=seq_X, array2=seq_Y)
