import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt

path = "sum_dataset.csv"
# path = "dataset_mood_smartphone.csv"

df = pd.read_csv(path, index_col=0)
df.info()
# print(df.head())

# for sum dataset
var_names = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen',
             'call', 'sms', 'appCat.builtin', 'appCat.communication',
             'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
             'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
             'appCat.utilities', 'appCat.weather']
df.drop('Unnamed: 0', axis=1, inplace=True)
df.info()
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
plt.show()
