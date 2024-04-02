import pandas as pd
from matplotlib import pyplot as plt
import pandas as pds
import numpy as np

dataset_df = pd.read_csv("dataset_mood_smartphone.csv")
dataset_df.info()


def get_all_rows_between(date1, time1, date2, time2):
    if date1:
        pass
