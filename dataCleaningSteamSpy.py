# standard library imports
import os
from ast import literal_eval
import itertools
import time
import re

# third-party imports
import numpy as np
import pandas as pd

from getSteamSpy import getSteamSpyFilesCSV

app_list_spy = getSteamSpyFilesCSV()

app_list_spy.drop('Unnamed: 0.1', inplace=True, axis=1)
app_list_spy.drop('Unnamed: 0', inplace=True, axis=1)

print('Rows:', app_list_spy.shape[0])
print('Columns:', app_list_spy.shape[1])


def process(df):
    df = df.copy()

    # handle missing values
    df = df[(df['name'].notnull()) & (df['name'] != 'none')]
    df = df[df['developer'].notnull()]
    df = df[df['price'].notnull()]

    # remove unwanted columns
    df = df.drop([
        'developer', 'publisher', 'score_rank', 'userscore', 'average_2weeks',
        'median_2weeks', 'price', 'initialprice', 'discount', 'ccu'
    ], axis=1)

    # reformat owners column
    df['owners'] = df['owners'].str.replace(',', '').str.replace(' .. ', '-')

    return df


steamspy_data = process(app_list_spy)

steamspy_data.to_csv(os.getcwd() + '/data/exports/steamspy_clean.csv', index=False)

steam_data = pd.read_csv(os.getcwd() + '/data/exports/steam_data_clean.csv')

merged = steam_data.merge(steamspy_data, left_on='steam_appid', right_on='appid', suffixes=('', '_steamspy'))

print(merged.head())

merged.to_csv(os.getcwd() + '/data/steam_+_spy_clean.csv', index=True)
