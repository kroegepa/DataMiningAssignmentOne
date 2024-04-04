from matplotlib import pyplot as plt
import pandas as pd

path = "dataset_mood_smartphone.csv"

analysis_df = pd.read_csv(path, index_col=0)

for name, group in analysis_df.groupby('variable', sort=False):
    plt.figure()
    group['value'].hist(alpha=0.75, bins=20)
    plt.title(f'Histogram of values for variable: {name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()