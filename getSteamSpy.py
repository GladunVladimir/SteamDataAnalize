import json
import time
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob

import steamspypi


def get_cooldown():
    cooldown = 70  # 1 minute plus a cushion

    return cooldown


def get_some_sleep():
    cooldown = get_cooldown()
    print("Sleeping for {} seconds on {}".format(cooldown, time.asctime()))

    time.sleep(cooldown)

    return


def download_a_single_page(page_no):
    print("Downloading steamSpy page={} on {}".format(page_no, time.asctime()))

    data_request = dict()
    data_request["request"] = "all"
    data_request["page"] = str(page_no)

    #data = steamspypi.download(data_request)

    data = pd.DataFrame.from_dict(steamspypi.download(data_request), orient='index')

    return data


def get_file_name(page_no):


    file_name = "steamspy_page_{}.csv".format(page_no)


    return file_name


def download_all_pages(num_pages):
    filePath = os.getcwd() + '/data/steamSpy/'

    for page_no in range(num_pages):
        file_name = get_file_name(page_no)

        if not Path(filePath + file_name).is_file():
            page_data = download_a_single_page(page_no=page_no)

            with open(filePath + file_name, "w", encoding="utf8") as f:
                page_data.to_csv(filePath + file_name)

            if page_no != (num_pages - 1):
                get_some_sleep()


def getSteamSpyFilesCSV():
    filePath = os.getcwd() + '/data/steamSpy/'

    if os.path.exists(filePath + 'steamSpyData_full.csv'):

        df_SteamSpy = pd.read_csv(filePath + 'steamSpyData_full.csv')


    else:
        csv_files = os.path.join(filePath, 'steamspy_page_*.csv')

        csv_files = glob.glob(csv_files)

        df_SteamSpy = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)

        Path(filePath + 'steamSpyData_full.csv').touch()

        df_SteamSpy.to_csv(filePath + 'steamSpyData_full.csv')

    return df_SteamSpy