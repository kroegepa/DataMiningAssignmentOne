import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
import scipy

# path = "sum_dataset.csv"
# path = "datasets/dataset_mood_smartphone.csv"
path = "datasets/per_day_participant_dataset_co_w_avg.csv"

df = pd.read_csv(path)
df.info()
# print(df.head())

# for sum dataset
var_names = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen',
             'call', 'sms', 'appCat.builtin', 'appCat.communication',
             'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office',
             'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown',
             'appCat.utilities', 'appCat.weather']

count_var_names = ["mood_count", "circumplex.arousal_count", "circumplex.valence_count", "activity_count",
                   "screen_count", "call_count", "sms_count", "appCat.builtin_count", "appCat.communication_count",
                   "appCat.entertainment_count", "appCat.finance_count", "appCat.game_count", "appCat.office_count",
                   "appCat.other_count", "appCat.social_count", "appCat.travel_count", "appCat.unknown_count",
                   "appCat.utilities_count", "appCat.weather_count"]

df = df.drop(count_var_names, axis=1)

df.drop(['id', 'date'], axis=1, inplace=True)
df.info()
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(method='kendall'), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig(f"graphs/correlation/corr_{path.split('/')[-1].split('.')[0]}.png")
plt.show()