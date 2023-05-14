import itertools
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use('default')
plt.rcdefaults()

curr_dir = os.getcwd()

df_requirements = pd.read_csv(curr_dir + '/data/exports/steam_requirements_data.csv')


def calc_rating(row):
    """Calculate rating score based on SteamDB method."""
    import math

    pos = row['positive']
    neg = row['negative']

    total_reviews = pos + neg
    if total_reviews != 0:
        average = pos / total_reviews

        #pulls score towards 50, pulls more strongly for games with few reviews
        score = average - (average * 0.5) * 2 ** (-math.log10(total_reviews + 1))

        return score * 100


def get_unique(series):
    """Get unique values from a Pandas series containing semi-colon delimited strings."""
    return set(list(itertools.chain(*series.apply(lambda x: [c for c in x.split(';')]))))


def process_cat_gen_tag(df):
    """Process categories, genres and steamspy_tags columns."""
    # get all unique category names
    cat_cols = get_unique(df['categories'])

    # only going to use these categories (can uncomment to use others)
    cat_cols = [
        # 'Local Multi-Player',
        # 'MMO',
        # 'Mods',
        'Multi-player',
        # 'Online Co-op',
        # 'Online Multi-Player',
        'Single-player'
    ]

    # create a new column for each category, with 1s indicating membership and 0s for non-members
    for col in sorted(cat_cols):
        col_name = re.sub(r'[\s\-\/]', '_', col.lower())
        col_name = re.sub(r'[()]', '', col_name)

        df[col_name] = df['categories'].apply(lambda x: 1 if col in x.split(';') else 0)

    # repeat for genre column names (get_unique used to find unique genre names,
    # not necessary but useful if keeping all of them)
    gen_cols = get_unique(df['genres'])

    # only keeping 'main' genres similar to steam store
    gen_cols = [
        # 'Accounting',
        'Action',
        'Adventure',
        # 'Animation & Modeling',
        # 'Audio Production',
        'Casual',
        # 'Design & Illustration',
        # 'Documentary',
        # 'Early Access',
        # 'Education',
        # 'Free to Play',
        # 'Game Development',
        # 'Gore',
        'Indie',
        'Massively Multiplayer',
        # 'Nudity',
        # 'Photo Editing',
        'RPG',
        'Racing',
        # 'Sexual Content',
        'Simulation',
        # 'Software Training',
        'Sports',
        'strategy'
        # 'Tutorial',
        # 'Utilities',
        # 'Video Production',
        # 'Violent',
        # 'Web Publishing'
    ]

    gen_col_names = []

    # create new columns for each genre with 1s for games of that genre
    for col in sorted(gen_cols):
        col_name = col.lower().replace('&', 'and').replace(' ', '_')
        gen_col_names.append(col_name)

        df[col_name] = df['genres'].apply(lambda x: 1 if col in x.split(';') else 0)
        # alternate method using np.where:
        # df[col_name] = np.where(df['genres'].str.contains(col), 1, 0)

    # remove "non-games" based on genre
    # if a row has all zeros in the new genre columns, it most likely isn't a game, so remove (mostly software)
    gen_sums = df[gen_col_names].sum(axis=1)
    df = df[gen_sums > 0].copy()

    # not using steamspy tags for now, as mostly overlap with genres
    # here's one way we could deal with them:
    # tag_cols = get_unique(df['steamspy_tags'])
    # df['top_tag'] = df['steamspy_tags'].apply(lambda x: x.split(';')[0])

    # remove redundant columns and return dataframe (keeping genres column for reference)
    df = df.drop(['categories'], axis=1)

    return df


def pre_process():
    """Preprocess Steam dataset for exploratory analysis."""
    df = pd.read_csv(curr_dir + '/data/steam_+_spy_clean.csv')

    # keep windows only, and remove platforms column
    df = df[df['platforms'].str.contains('windows')].drop('platforms', axis=1).copy()

    # keep lower bound of owners column, as integer
    df['owners'] = df['owners'].str.split('-').apply(lambda x: x[0]).astype(int)

    # calculate rating, as well as simple ratio for comparison
    df['total_ratings'] = df['positive'] + df['negative']
    df['rating_ratio'] = df['positive'] / df['total_ratings']
    df['rating'] = df.apply(calc_rating, axis=1)

    # convert release_date to datetime type and create separate column for release_year
    df['release_date'] = df['release_date'].astype('datetime64[ns]')
    df['release_year'] = df['release_date'].apply(lambda x: x.year)

    # process genres, categories and steamspy_tag columns
    df = process_cat_gen_tag(df)

    return df


data = pre_process()

warnings.filterwarnings('ignore')

# Create a column to split free vs paid games
data['type'] = 'Free'
data.loc[data['price'] > 0, 'type'] = 'Paid'

# ensure no 0s in columns we're applying log to
df = data[(data['owners'] > 0) & (data['total_ratings'] > 0)].copy()

eda_df = pd.DataFrame(zip(
    # df['rating'],
    # np.log10(df['total_ratings']),
    # np.log10(df['owners']),
    df.price,
    df['release_year'],
    df['type'],
),
    columns=[
        # 'Rating Score',
        # 'Total Ratings (log)',
        # 'Owners (log)',
        'Current Price',
        'Release Year',
        'Type'
    ])

sns.pairplot(eda_df, hue='Type', palette=['blue', 'gray'])

if not os.path.isdir('plot'):
    os.mkdir(curr_dir + '/plot/')
plt.savefig(curr_dir + '/plot/PaidAndFree.png')
plt.show()

df = data.copy()

years = []
lt_20k = []
gt_20k = []

for year in sorted(df['release_year'].unique()):
    if year < 2006 or year > 2022:
        # very few releases in data prior to 2006, and we're still in 2022 (at time of writing)
        # so ignore these years
        continue

    # subset dataframe by year
    year_df = df[df.release_year == year]

    # calculate total with less than 20,000 owners, and total with 20,000 or more
    total_lt_20k = year_df[year_df.owners < 20000].shape[0]
    total_gt_20k = year_df[year_df.owners >= 20000].shape[0]

    years.append(year)
    lt_20k.append(total_lt_20k)
    gt_20k.append(total_gt_20k)

owners_df = pd.DataFrame(zip(years, lt_20k, gt_20k),
                         columns=['year', 'Under 20,000 Owners', '20,000+ Owners'])

ax = owners_df.plot(x='year', y=[1, 2], kind='bar', stacked=True, color=['tab:blue', 'gray'])

ax.set_xlabel('')
ax.set_ylabel('Number of Releases')
ax.set_title('Number of releases by year, broken down by number of owners')
sns.despine()

plt.savefig(curr_dir + '/plot/NumberOfReleases.png')
plt.show()

top_ten = df.sort_values(by='rating', ascending=False).head(10)

# storing category and genre columns in a variable, as we'll be accessing them often
cat_gen_cols = df.columns[-13:-1]
ax = top_ten[cat_gen_cols].sum().plot.bar(figsize=(8, 5))

ax.fill_between([-.5, 1.5], 10, alpha=.2)
ax.text(0.5, 9.1, 'Categories', fontsize=11, color='tab:blue', alpha=.8, horizontalalignment='center')

ax.set_ylim([0, 9.5])
ax.set_ylabel('Count')
ax.set_title('Frequency of categories and genres in top ten games')

plt.tight_layout()
plt.savefig(curr_dir + '/plot/FreqTop.png')
plt.show()

gen_cols = ['action',
            'adventure',
            'casual',
            'indie',
            'massively_multiplayer',
            'rpg',
            'racing',
            'simulation',
            'sports',
            'strategy']


def plot_owners_comparison(df):
    # percentage of games in each genre
    total_owners_per_genre = df[gen_cols].multiply(df['owners'], axis='index').sum()
    average_owners_per_genre = total_owners_per_genre / df[gen_cols].sum()

    fig, ax1 = plt.subplots(figsize=(13, 7))

    color = 'tab:gray'
    (df[gen_cols].mean() * 100).sort_index(ascending=False).plot.barh(ax=ax1, color=color, alpha=.9, position=1,
                                                                      fontsize=14, width=0.4)
    # ax1.set_ylabel('genre')

    ax1.set_xlabel('% of games (creation popularity)', color=color, size=12)
    ax1.tick_params(axis='x', labelcolor=color)
    ax1.tick_params(axis='y', left='off', top='off')
    # ax1.axes.get_yaxis().set_visible(False)

    ax2 = ax1.twiny()

    color = 'tab:blue'
    average_owners_per_genre.sort_index(ascending=False).plot.barh(ax=ax2, color=color, alpha=1, position=0,
                                                                   fontsize=14, width=0.4)
    ax2.set_xlabel('Average owners per game (consumer popularity)', color=color, size=12)
    ax2.tick_params(axis='x', labelcolor=color)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_ylim([-.5, 9.5])

    plt.tight_layout()
    plt.savefig(curr_dir + '/plot/Owners.png')
    plt.show()


plot_owners_comparison(df[df.owners <= 10000000])

g_df = pd.DataFrame()

for col in gen_cols:
    temp_df = df[df[col] == 1].copy()
    temp_df['genre'] = col
    g_df = pd.concat([g_df, temp_df], axis=0)

recent_df = g_df[g_df['release_year'] >= 2022].copy()
ax = sns.stripplot(x='price', y='genre', data=recent_df, jitter=True, alpha=.5, linewidth=1, palette=['red'])
plt.tight_layout()
plt.savefig(curr_dir + '/plot/Price.png')
plt.show()

df[df.publisher == 'Ubisoft'][gen_cols].mean().plot.bar(figsize=(10, 8), color='tab:blue')
plt.title('Proportion of games released by Ubisoft in each genre')
plt.tight_layout()
plt.savefig(curr_dir + '/plot/Ubisoft.png')
plt.show()

df[df.publisher == 'SEGA'][gen_cols].mean().plot.bar(figsize=(10, 8), color='tab:blue')
plt.title('Proportion of games released by SEGA in each genre')
plt.tight_layout()
plt.savefig(curr_dir + '/plot/SEGA.png')
plt.show()

df[df.publisher == 'Electronic Arts'][gen_cols].mean().plot.bar(figsize=(10, 8), color='tab:blue')
plt.title('Proportion of games released by Electronic Arts in each genre')
plt.tight_layout()
plt.savefig(curr_dir + '/plot/Electronic_Arts.png')
plt.show()

# print(df_requirements['mac_requirements'].where(df_requirements['mac_requirements'] == '[]').sum())

fig = plt.figure(figsize=(10, 6))

dfa = data[data.owners >= 20000].copy()
dfa['subset'] = '20,000+ Owners'

dfb = data.copy()
dfb['subset'] = 'All Data'

ax = sns.boxplot(x='subset', y='rating', hue='type', data=pd.concat([dfa, dfb]))

ax.set(xlabel='', ylabel='Rating (%)')
plt.show()

# paid with over 20,000 owners
df = data[(data.owners >= 20000) & (data.price > 0)].copy()

fig, axarr = plt.subplots(1, 2, figsize=(10, 5))

df.rating.plot.hist(range=(0, 100), bins=10, ax=axarr[0], alpha=.8)
# sns.distplot(df.rating, bins=range(0,100,10), ax=axarr[0])

# plot line for mean on histogram
mean = df.rating.mean()
axarr[0].axvline(mean, c='black', alpha=.6)
axarr[0].text(mean - 1, 1600, f'Mean: {mean:.0f}%', c='black', ha='right', alpha=.8)

ax = sns.boxplot(x='rating', data=df, orient='v', ax=axarr[1])
fig.suptitle('Distribution of rating scores for paid games with 20,000+ owners')
plt.show()
