# WE WILL LEARN COMPLETE EDA WITH AN REAL WORLD DATASET EXAMPLE
# WE WILL USE insurance.csv FILE FOR THIS PROJECT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

"""
The warnings module is used to control how Python displays warning messages.
Sometimes libraries (especially in data science like NumPy, Pandas, Scikit-learn, TensorFlow, etc.) generate warnings that don't stop your code but clutter the output.
"warnings.filterwarnings('ignore')" tells Python to hide all warning messages.
"""

df = pd.read_csv('insurance.csv')
print(df)

# EDA

# analysis
print(df.shape)             # returns the number of rows and colm that are present
print(df.head())            # returns the first five record
print(df.info())            # provides information about the dataframe
print(df.describe())        # provides description of the dataframe
print(df.isnull().sum())    # provides number of null values in each column
print(df.columns)           # returns all the columns

numeric_columns = ['age', 'bmi', 'children', 'charges']

for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde = True, bins = 20)
    plt.show()

sns.countplot(x = df['children'])
plt.show()

sns.countplot(x = df['sex'])
plt.show()

sns.countplot(x = df['smoker'])
plt.show()

for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x = df[col])
    plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only = True),annot=True)
plt.show()


# Data Cleaning and Preprocessing
df_cleaned = df.copy()
df_cleaned.drop_duplicates(inplace=True)
print(df_cleaned.isnull().sum())
print(df_cleaned.dtypes)
print(df_cleaned['sex'].value_counts())         # shows the number of males and females
df_cleaned['sex'] = df_cleaned['sex'].map({"male":0,"female":1})
print(df_cleaned.dtypes)

print(df_cleaned['smoker'].value_counts())
df_cleaned['smoker'] = df_cleaned['smoker'].map({"no":0,"yes":1})
print(df_cleaned.dtypes)

df_cleaned.rename(columns = {'sex':'is_female','smoker':'is_smoker'},inplace=True)
print(df_cleaned.head())

print(df_cleaned['region'].value_counts())

"""
we have done label encoding for sex and smoker
but since there are more than 2 values for region, here we will perform one-hot encoding
region_southeast,region_southwest,region_northwest,region_northeast
"""

df_cleaned = pd.get_dummies(df_cleaned,columns=['region'])
print(df_cleaned.head())

df_cleaned = df_cleaned.astype(int)     # converts all dtypes to int
print(df_cleaned)


# Feature Engineering and Extraction (Selection)

sns.histplot(df_cleaned['bmi'])
plt.show()


df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins = [0,18.5,24.9,29.9,float('inf')],
    labels = ['Underweight','Normal','Overweight','Obese']
)

df_cleaned = pd.get_dummies(df_cleaned,columns=['bmi_category'])
df_cleaned = df_cleaned.astype(int)
print(df_cleaned.tail())            # shows the last 5 records


from sklearn.preprocessing import StandardScaler
cols=['age','bmi','children']
scaler = StandardScaler()
df_cleaned[cols]=scaler.fit_transform(df_cleaned[cols])
print(df_cleaned.head())

from scipy.stats import pearsonr

# List of features to check against target
selected_features = [
    'age', 'bmi', 'children', 'is_female', 'is_smoker', 'region_northeast',
    'region_northwest', 'region_southeast', 'region_southwest', 'bmi_category_Underweight',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_features
}

correlation_df = pd.DataFrame(list(correlations.items()),columns=['Feature','Pearson Correlation'])

print(correlation_df.sort_values(by='Pearson Correlation',ascending = False))

"""
correlation lies between -1 to 1
"""

"""chi square test"""

cat_features = [
    'is_female', 'is_smoker','region_northeast',
    'region_northwest', 'region_southeast', 'region_southwest','bmi_category_Underweight',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

from scipy.stats import chi2_contingency
alpha = 0.05

df_cleaned['charges_bin'] = pd.qcut(df_cleaned['charges'],q=4,labels=False)
chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
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

final_df = df_cleaned[['age', 'is_female', 'bmi', 'children', 'is_smoker', 'charges', 'region_southeast', 'bmi_category_Obese']]

## now using this dataframe we will create a ML model



















