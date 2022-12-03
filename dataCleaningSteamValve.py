import csv
import datetime as dt
import json
import os
import statistics
import time
import re
from ast import literal_eval
from ssl import SSLError
import numpy as np
import pandas as pd
import requests


from getSteamValve import *


app_list_valve = getSteamValveFilesCSV(steam_columns)



app_list_valve.drop('Unnamed: 0.1', inplace=True, axis=1)
app_list_valve.drop('Unnamed: 0', inplace=True, axis=1)


# print(app_list_valve.iloc[0])

print('Rows:', app_list_valve.shape[0])
print('Columns:', app_list_valve.shape[1])


def drop_null_cols(df, thresh=0.5):
    """Drop columns with more than a certain proportion of missing values (Default 50%)."""
    cutoff_count = len(df) * thresh

    return df.dropna(thresh=cutoff_count, axis=1)


def process_name_type(df):
    """Remove null values in name and type columns, and remove type column."""
    df = df[df['type'].notnull()]

    df = df[df['name'].notnull()]
    df = df[df['name'] != 'none']

    df = df.drop('type', axis=1)

    return df


def process(df):
    """Process data set. Will eventually contain calls to all functions we write."""

    # Copy the input dataframe to avoid accidentally modifying original data
    df = df.copy()

    # Remove duplicate rows - all appids should be unique
    df = df.drop_duplicates()

    # Remove collumns with more than 50% null values
    df = drop_null_cols(df)

    # Process rest of columns
    df = process_name_type(df)

    return df


def process_age(df):
    """Format ratings in age column to be in line with the PEGI Age Ratings system."""
    # PEGI Age ratings: 3, 7, 12, 16, 18
    cut_points = [-1, 0, 3, 7, 12, 16, 2000]
    label_values = [0, 3, 7, 12, 16, 18]
    df['required_age'] = df['required_age'].replace(['18+'], '18')
    df['required_age'] = df['required_age'].replace(['21+'], '21')

    df['required_age'] = pd.to_numeric(df['required_age'])

    df['required_age'] = pd.cut(df['required_age'], bins=cut_points, labels=label_values)

    return df


def process_platforms(df):
    """Split platforms column into separate boolean columns for each platform."""
    # evaluate values in platforms column, so can index into dictionaries
    df = df.copy()

    def parse_platforms(x):
        d = literal_eval(x)

        return ';'.join(platform for platform in d.keys() if d[platform])

    df['platforms'] = df['platforms'].apply(parse_platforms)

    return df

# not_free_and_null_price = platforms_df[(platforms_df['is_free'] == False) & (platforms_df['price_overview'].isnull())]

def print_steam_links(df):
    """Print links to store page for apps in a dataframe."""
    url_base = "https://store.steampowered.com/app/"

    for i, row in df.iterrows():
        appid = row['steam_appid']
        name = row['name']

        print(name + ':', url_base + str(appid))


class RealTimeCurrencyConverter():
    def __init__(self,url):
        self.data= requests.get(url).json()
        self.currencies = self.data['rates']

    def convert(self, from_currency, to_currency, amount):
        initial_amount = amount
        # first convert it into USD if it is not in USD.
        # because our base currency is USD
        if from_currency != 'USD':
            amount = amount / self.currencies[from_currency]

            # limiting the precision to 4 decimal places
        amount = round(amount * self.currencies[to_currency], 4)
        return amount


def process_price(df):
    url = 'https://api.exchangerate-api.com/v4/latest/TRY'
    converter = RealTimeCurrencyConverter(url)
    """Process price_overview column into formatted price column, and take care of package columns."""
    df = df.copy()

    def parse_price(x):
        if x is not np.nan:
            return literal_eval(x)
        else:
            return {'currency': 'TRY', 'initial': -1}

    # evaluate as dictionary and set to -1 if missing
    df['price_overview'] = df['price_overview'].apply(parse_price)

    # create columns from currency and initial values
    df['currency'] = df['price_overview'].apply(lambda x: x['currency'])
    df['price'] = df['price_overview'].apply(lambda x: x['initial'])

    # set price of free games to 0
    df.loc[df['is_free'], 'price'] = 0

    # remove non-TRY rows
    df = df[df['currency'] == 'TRY']

    # remove rows where price is -1
    df = df[df['price'] != -1]

    # change price to display in pounds (can apply to all now -1 rows removed)
    df.loc[df['price'] > 0, 'price'] /= 100
    df.loc[df['price'] > 0, 'price'] = converter.convert('TRY','USD', df.loc[df['price'] > 0, 'price'])

    # remove columns no longer needed
    df = df.drop(['is_free', 'currency', 'price_overview', 'packages', 'package_groups'], axis=1)

    return df


def process_language(df):
    """Process supported_languages column into a boolean 'is english' column."""
    df = df.copy()

    # drop rows with missing language data
    df = df.dropna(subset=['supported_languages'])

    df['english'] = df['supported_languages'].apply(lambda x: 1 if 'english' in x.lower() else 0)
    df['russian'] = df['supported_languages'].apply(lambda x: 1 if 'russian' in x.lower() else 0)
    df = df.drop('supported_languages', axis=1)

    return df


def process_developers_and_publishers(df):
    """Parse columns as semicolon-separated string."""
    # remove rows with missing data (~ means not)
    df = df[(df['developers'].notnull()) & (df['publishers'] != "['']")].copy()
    df = df[~(df['developers'].str.contains(';')) & ~(df['publishers'].str.contains(';'))]
    df = df[(df['publishers'] != "['NA']") & (df['publishers'] != "['N/A']")]

    # create list for each
    df['developer'] = df['developers'].apply(lambda x: ';'.join(literal_eval(x)))
    df['publisher'] = df['publishers'].apply(lambda x: ';'.join(literal_eval(x)))

    df = df.drop(['developers', 'publishers'], axis=1)

    return df


def process_categories_and_genres(df):
    df = df.copy()
    df = df[(df['categories'].notnull()) & (df['genres'].notnull())]

    for col in ['categories', 'genres']:
        df[col] = df[col].apply(lambda x: ';'.join(item['description'] for item in literal_eval(x)))

    return df


def process_achievements_and_descriptors(df):
    """Parse as total number of achievements."""
    df = df.copy()

    df = df.drop('content_descriptors', axis=1)

    def parse_achievements(x):
        if x is np.nan:
            # missing data, assume has no achievements
            return 0
        else:
            # else has data, so can extract and return number under total
            return literal_eval(x)['total']

    df['achievements'] = df['achievements'].apply(parse_achievements)

    return df


def start_process(df):
    """Process data set. Will eventually contain calls to all functions we write."""

    # Copy the input dataframe to avoid accidentally modifying original data
    df = df.copy()

    # Remove duplicate rows - all appids should be unique
    df = df.drop_duplicates()

    # Remove collumns with more than 50% null values
    df = drop_null_cols(df)

    # Process columns
    df = process_name_type(df)
    df = process_age(df)
    df = process_platforms(df)
    df = process_price(df)
    df = process_language(df)
    df = process_developers_and_publishers(df)
    df = process_categories_and_genres(df)
    df = process_achievements_and_descriptors(df)

    return df


def export_data(df, filename):
    """Export dataframe to csv file, filename prepended with 'steam_'.

    filename : str without file extension
    """
    filepath = os.getcwd() + '/data/exports/steam_' + filename + '.csv'

    df.to_csv(filepath, index=False)

    print_name = filename.replace('_', ' ')
    print("Exported {} to '{}'".format(print_name, filepath))


def process_descriptions(df, export=False):
    """Export descriptions to external csv file then remove these columns."""
    # remove rows with missing description data
    df = df[df['detailed_description'].notnull()].copy()

    # remove rows with unusually small description
    df = df[df['detailed_description'].str.len() > 20]

    # by default we don't export, useful if calling function later
    if export:
        # create dataframe of description columns
        description_data = df[['steam_appid', 'detailed_description', 'about_the_game', 'short_description']]

        export_data(description_data, filename='description_data')

    # drop description columns from main dataframe
    df = df.drop(['detailed_description', 'about_the_game', 'short_description'], axis=1)

    return df


def process_media(df, export=False):
    """Remove media columns from dataframe, optionally exporting them to csv first."""
    df = df[df['screenshots'].notnull()].copy()

    if export:
        media_data = df[['steam_appid', 'header_image', 'screenshots', 'background', 'movies']]

        export_data(media_data, 'media_data')

    df = df.drop(['header_image', 'screenshots', 'background', 'movies'], axis=1)

    return df


def process_info(df, export=False):
    """Drop support information from dataframe, optionally exporting beforehand."""
    if export:
        support_info = df[['steam_appid', 'website', 'support_info']].copy()

        support_info['support_info'] = support_info['support_info'].apply(lambda x: literal_eval(x))
        support_info['support_url'] = support_info['support_info'].apply(lambda x: x['url'])
        support_info['support_email'] = support_info['support_info'].apply(lambda x: x['email'])

        support_info = support_info.drop('support_info', axis=1)

        # only keep rows with at least one piece of information
        support_info = support_info[(support_info['website'].notnull()) | (support_info['support_url'] != '') | (
                    support_info['support_email'] != '')]

        export_data(support_info, 'support_info')

    df = df.drop(['website', 'support_info'], axis=1)

    return df


def process_requirements(df, export=False):
    if export:
        requirements = df[['steam_appid', 'pc_requirements', 'mac_requirements', 'linux_requirements']].copy()

        # remove rows with missing pc requirements
        requirements = requirements[requirements['pc_requirements'] != '[]']

        requirements['requirements_clean'] = (requirements['pc_requirements']
                                              .str.replace(r'\\[rtn]', '')
                                              .str.replace(r'<[pbr]{1,2}>', ' ')
                                              .str.replace(r'<[\/"=\w\s]+>', '')
                                              )

        requirements['requirements_clean'] = requirements['requirements_clean'].apply(lambda x: literal_eval(x))

        # split out minimum and recommended into separate columns
        requirements['minimum'] = requirements['requirements_clean'].apply(
            lambda x: x['minimum'].replace('Minimum:', '').strip() if 'minimum' in x.keys() else np.nan)
        requirements['recommended'] = requirements['requirements_clean'].apply(
            lambda x: x['recommended'].replace('Recommended:', '').strip() if 'recommended' in x.keys() else np.nan)

        requirements = requirements.drop('requirements_clean', axis=1)

        export_data(requirements, 'requirements_data')

    df = df.drop(['pc_requirements', 'mac_requirements', 'linux_requirements'], axis=1)

    return df


def process_release_date(df):
    df = df.copy()

    def eval_date(x):
        x = literal_eval(x)
        if x['coming_soon']:
            return ''  # return blank string so can drop missing at end
        else:
            return x['date']

    df['release_date'] = df['release_date'].apply(eval_date)

    def parse_date(x):
        if re.search(r'[\d]{1,2} [A-Za-z]{3}, [\d]{4}', x):
            return x.replace(',', '')
        elif re.search(r'[A-Za-z]{3} [\d]{4}', x):
            return '1 ' + x
        elif x == '':
            return np.nan
        else:
            # Should be everything, print out anything left just in case
            print(x)

    df['release_date'] = df['release_date'].apply(parse_date)
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d %b %Y', errors='coerce')

    df = df[df['release_date'].notnull()]

    return df


def process(df):
    """Process data set. Will eventually contain calls to all functions we write."""

    # Copy the input dataframe to avoid accidentally modifying original data
    df = df.copy()

    # Remove duplicate rows - all appids should be unique
    df = df.drop_duplicates()

    # Remove collumns with more than 50% null values
    df = drop_null_cols(df)

    # Process columns
    df = process_name_type(df)
    df = process_age(df)
    df = process_platforms(df)
    df = process_price(df)
    df = process_language(df)
    df = process_developers_and_publishers(df)
    df = process_categories_and_genres(df)
    df = process_achievements_and_descriptors(df)
    df = process_release_date(df)

    # Process columns which export data
    df = process_descriptions(df, export=False)
    df = process_media(df, export=False)
    df = process_info(df, export=False)
    df = process_requirements(df, export=False)

    return df

steam_data = process(app_list_valve)
print(steam_data.isnull().sum())

print(app_list_valve.info(verbose=False, memory_usage="deep"))


print(steam_data.info(verbose=False, memory_usage="deep"))

steam_data.to_csv(os.getcwd() + '/data/exports/steam_data_clean.csv', index=False)