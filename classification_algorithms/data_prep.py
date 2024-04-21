import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# parameters
show_histo = True
show_output = True
show_extended_output = True
normalize = True
output_name = "dataset_xgboost.npz"

seq_length = 1

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
df["date_temp"] = pd.to_datetime(df["date"], dayfirst=True)
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

# drop useless columns
df = df.drop(columns=["date_temp", "batch", "mood_count"])
if show_output:
    print(df.columns)

if normalize:
    # normalize data
    for col in tqdm(df.columns, desc="normalizing columns"):
        if col not in ["mood_class", "date", "id"]:
            # fit and transform the data using MinMaxScaler
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])

seq_X = []
seq_Y = []
if seq_length != 1:
    # create sequences
    for name, group_id in df.groupby("id", sort=False):
        # check if there are enough rows for sequence
        if len(group_id) > seq_length:
            # for each participant take the first seq_length rows then do it +1
            for index in range(len(group_id) - seq_length - 1):
                seq_X.append(group_id.iloc[index:index + seq_length].drop(columns=["date", "id"]).to_numpy())
                seq_Y.append(group_id.iloc[index + seq_length + 1].drop(columns=["date", "id"]).to_numpy()[-1])

else:
    df_dropped = df.drop(columns=["date", "id"])
    for index in range(len(df_dropped) - 1):  # -1 to prevent out-of-bounds in seq_Y
        seq_X.append(df_dropped.iloc[index].to_numpy())
        seq_Y.append(df_dropped.iloc[index + 1].to_numpy()[-1])  # Target from next row

if show_output:
    print(seq_X)
    print(seq_Y)

np.savez(output_name, array1=seq_X, array2=seq_Y)
