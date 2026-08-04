## Football Player Market Value Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv('footballers.csv')
print(df)

# analysis
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
print(df.duplicated().sum())

print(df['Nationality'].nunique())
print(df['Nationality'].value_counts())

print(df['Club'].nunique())
print(df['Club'].value_counts())


print(df['Preferred Positions'].value_counts())

print(df['Wage'].head())
print(df['Wage'].dtype)

print(df['Value'].head())       # Target analysis
print(df['Value'].dtype)

print(df.isnull().sum())


def convert_money(x):
    x = x.replace('€','')

    if 'M' in x:
        return float(x.replace('M',''))*1000000
    elif 'K' in x:
        return float(x.replace('K',''))*1000
    else:
        return float(x)

df['Value'] = df['Value'].apply(convert_money)
df['Wage'] = df['Wage'].apply(convert_money)

"""
def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde = True)


plotting('Age',1)
plotting('Overall',2)
plotting('Wage',3)
plotting('Value',4)
plt.show()
"""
df['Elite'] = (df['Overall'] >= 90).astype(int)     # Feature Engineering

print(df[df['Elite'] == 1]['Name'])
print(df[df['Elite'] == 1]['Nationality'].value_counts())
print(df[df['Elite'] == 1]['Club'].value_counts())



for col in df.columns:
    print(f"{col}: {df[col].isnull().sum()}")       # shows which colm have how many null

position_cols = [
    'CAM','CB','CDM','CF','CM','LAM','LB','LCB','LCM','LDM',
    'LF','LM','LS','LW','LWB','RAM','RB','RCB','RCM','RDM',
    'RF','RM','RS','RW','RWB','ST'
]                                                            # fills the null position. now only the clubs colm have null value

df[position_cols] = df[position_cols].fillna(0)

df['Club'] = df['Club'].fillna('Free Agent')            # no null clubs

for col in df.columns:
    print(f"{col}: {df[col].isnull().sum()}")       # Checking again

# converting to numeric values
exclude = ['Name', 'Nationality', 'Club', 'Preferred Positions']

"""
we have found that some places instead of 11-10 it is made 11-Oct. so have to change that into 11-10.
"""
month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

def convert_date(x):

    x = str(x).strip()

    if '-' in x:
        parts = x.split('-')

        if len(parts) == 2 and parts[1] in month_map:
            return abs(int(parts[0]) - month_map[parts[1]])

    return x

for col in df.columns:
    if col not in exclude:
        df[col] = df[col].apply(convert_date)

def convert_position(x):

    x = str(x).strip()

    if '+' in x:
        a, b = x.split('+')
        return int(a) + int(b)

    elif '-' in x:
        a, b = x.split('-')
        return abs(int(a) - int(b))

    return x

for col in df.columns:
    if col not in exclude:
        df[col] = df[col].apply(convert_position)

for col in df.columns:
    if col not in exclude:
        df[col] = pd.to_numeric(df[col]).astype(int)

# grouping features
profile = ['Age', 'Overall', 'Potential', 'Special','Elite']
physical = ['Acceleration', 'Sprint speed', 'Agility', 'Balance', 'Jumping', 'Stamina', 'Strength']
control = ['Ball control', 'Dribbling', 'Composure', 'Reactions']
passing = ['Vision', 'Short passing', 'Long passing', 'Crossing', 'Curve', 'Free kick accuracy']
attacking = ['Finishing', 'Shot power', 'Long shots', 'Positioning', 'Volleys', 'Heading accuracy', 'Penalties']
defending = ['Aggression', 'Interceptions', 'Marking', 'Standing tackle', 'Sliding tackle']
goalkeeping = ['GK diving', 'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes']
position_rating = ['CAM', 'CB', 'CDM', 'CF', 'CM', 'LAM', 'LB', 'LCB', 'LCM', 'LDM',
                   'LF', 'LM', 'LS', 'LW', 'LWB', 'RAM', 'RB', 'RCB', 'RCM', 'RDM',
                   'RF', 'RM', 'RS', 'RW', 'RWB', 'ST']

# One hot encoding
# Create binary columns
positions = [
    'GK', 'CB', 'LB', 'RB', 'LWB', 'RWB','CDM', 'CM', 'CAM','LM', 'RM','LW', 'RW',
    'LF', 'RF','CF','ST','LS', 'RS','LCB', 'RCB','LCM', 'RCM','LDM', 'RDM','LAM', 'RAM'
]

for pos in positions:
    df[f'{pos}_preferred_position'] = (
        df['Preferred Positions']
        .str.contains(rf'\b{pos}\b', regex=True)
        .astype(int)
    )

# Drop the original column
df.drop(columns='Preferred Positions', inplace=True)

# grouping the one hot encoded colms
preferred_position = [f'{pos}_preferred_position' for pos in positions]

selected_features = (
    profile +
    physical +
    control +
    passing +
    attacking +
    defending +
    goalkeeping +
    position_rating +
    preferred_position
)
print(len(selected_features))
print(selected_features)
"""
for feature in selected_features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[feature], y=df['Value'])
    plt.title(f'{feature} vs Value')
    plt.show()
"""

# Pearson Correlation
from scipy.stats import pearsonr

# List of features to check against target
correlations = {
    feature: pearsonr(df[feature], df['Value'])[0]
    for feature in selected_features
}

correlation_df = pd.DataFrame(
    list(correlations.items()),
    columns=['Feature', 'Pearson Correlation']
)

print(correlation_df.sort_values(by='Pearson Correlation', ascending=False))


# Dropping numerical features
drop_features = correlation_df[
    abs(correlation_df['Pearson Correlation']) < 0.10
]['Feature'].tolist()

print(drop_features)

df.drop(columns=drop_features, inplace=True)

"""chi square test"""

preferred_position = [
    feature for feature in preferred_position
    if feature in df.columns
]

cat_features = ['Elite', 'Nationality', 'Club'] + preferred_position

from scipy.stats import chi2_contingency
alpha = 0.05

df['Value_bin'] = pd.qcut(df['Value'], q=4, labels=False)
chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df[col], df['Value_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Accept Null (Drop Feature)'

    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value')

print(chi2_df)

## Droping categorical features
drop_cat = chi2_df[
    chi2_df['Decision'] == 'Accept Null (Drop Feature)'
].index.tolist()

print(drop_cat)

df.drop(columns=drop_cat, inplace=True)
df.drop(columns='Value_bin', inplace=True)

selected_features = [
    col for col in selected_features
    if col in df.columns
]
# scaling
from sklearn.preprocessing import StandardScaler

cols = [
    col for col in selected_features
    if df[col].nunique() > 2
]

scaler = StandardScaler()
df[cols]=scaler.fit_transform(df[cols])

final_df = df.copy()
print(final_df.head())
