from matplotlib import pyplot as plt
import pandas as pd
from os.path import exists
from os import mkdir

path = "dataset_mood_smartphone.csv"
save_path = "./graphs"
analysis_df = pd.read_csv(path, index_col=0)

doHisto = True
doBoxplot = True

if doHisto:
    if not exists(save_path+"/histograms"):
        mkdir(save_path+"/histograms")

    for name, group in analysis_df.groupby('variable', sort=False):
        plt.figure()
        group['value'].hist(alpha=0.75, bins=20)
        plt.title(f'Histogram of values for variable: {name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(False)
        plt.savefig(save_path+"/histograms/"+f"histo_{name}.png")
        plt.show()

if doBoxplot:
    if not exists(save_path+"/boxplots"):
        mkdir(save_path+"/boxplots")

    for name, group in analysis_df.groupby('variable', sort=False):
        plt.figure()
        group['value'].plot.box()
        plt.title(f'Boxplot of values for variable: {name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(False)
        plt.savefig(save_path+"/boxplots/" + f"histo_{name}.png")
        plt.show()
