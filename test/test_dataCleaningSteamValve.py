import pandas as pd
import datatest as dt


def setUpModule():
    global df
    with dt.working_directory('/Users/vladimirgl/PycharmProjects/SteamDataAnalize/data/exports'):
        df = pd.read_csv('/Users/vladimirgl/PycharmProjects/SteamDataAnalize/data/exports/steam_data_clean.csv')

# Проверка на наличие столбцов в таблице
class TestMovies(dt.DataTestCase):
    @dt.mandatory
    def test_columns(self):
        self.assertValid(
            df.columns,
            {'name',
             'achievements',
             'categories',
             'developer',
             'english',
             'genres',
             'platforms',
             'price',
             'publisher',
             'release_date',
             'required_age',
             'russian',
             'steam_appid'},
        )
# Проверка типов данных в столбце “name”
    def test_type_name(self):
        self.assertValidRegex(df['name'], str)
# Проверка типов данных в столбце “achievements”
    def test_type_achievements(self):
        self.assertValidRegex(df['achievements'], int)
# Проверка типов данных в столбце “categories”
    def test_type_categories(self):
        self.assertValidRegex(df['categories'], str)
# Проверка типов данных в столбце “developer”
    def test_type_developer(self):
        self.assertValidRegex(df['developer'], str)
# Проверка типов данных в столбце “english_column”
    def test_type_english_column(self):
        self.assertValidRegex(df['english'], int)
# Проверка типов данных в столбце “genres”
    def test_type_genres(self):
        self.assertValidRegex(df['genres'], str)
# Проверка типов данных в столбце “platforms”
    def test_type_platforms(self):
        self.assertValidRegex(df['platforms'], str)
# Проверка типов данных в столбце “price”
    def test_type_price(self):
        self.assertValidRegex(df['price'], float)
# Проверка типов данных в столбце “publisher”
    def test_type_publisher(self):
        self.assertValidRegex(df['publisher'], object)
# Проверка типов данных в столбце “release_date”
    def test_type_release_date(self):
        self.assertValidRegex(df['release_date'], str)
# Проверка типов данных в столбце “required_age”
    def test_type_required_age(self):
        self.assertValidRegex(df['required_age'], int)
# Проверка типов данных в столбце “russian_column”
    def test_type_russian_column(self):
        self.assertValidRegex(df['russian'], int)
# Проверка типов данных в столбце “steam_appid”
    def test_type_steam_appid(self):
        self.assertValidRegex(df['steam_appid'], int)

# Проверка на допустимые значения в ячейке “platforms”
    def test_var_platforms(self):
        self.assertValidSuperset(
            df['platforms'],
            {'windows', 'mac', 'windows;mac', 'windows;mac;linux'},
        )
# Проверка на допустимые значения в ячейке “categories”
    def test_var_categories(self):
        self.assertValidSuperset(
            df['categories'],
            {'Single-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Cloud;Steam Leaderboards;Remote Play on TV'},
        )
# Проверка на допустимые значения в ячейке “required_age
    def test_var_required_age(self):
        self.assertValidSuperset(
            df['required_age'],
            {0, 3, 7, 12, 16, 18}
        )
# Проверка на допустимые значения в ячейке “genres”
    def test_var_genres(self):
        self.assertValidSuperset(
            df['genres'],
            {'Action;Adventure;Indie',
             'Indie;Strategy',
             'Adventure;Casual;Indie',
             'Sexual Content;Nudity;Violent;Adventure;Casual;Indie;Simulation',
             'Indie;Simulation;Design & Illustration;Education;Web Publishing'}
        )