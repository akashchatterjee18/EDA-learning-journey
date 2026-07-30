import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv('heart.csv')
print(df)

# EDA

# analysis
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
print(df.duplicated().sum())                    # returns no of duplicate values
print(df['HeartDisease'].value_counts())        # returns no of people with and withour heart diesease
#df['HeartDisease'].value_counts().plot(kind = 'bar')       # to plot the value
#plt.show()

print(df.isnull().sum())

def plotting(var,num):
    plt.subplot(2,2,num)
    sns.histplot(df[var],kde = True)
    plt.show()


plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)


print(df['Cholesterol'].value_counts())     # 172 people have cholesterol of zero, which isnt possible



# Data Cleaning along with EDA
'''cholesterol cleaning'''
ch_mean = df.loc[df['Cholesterol']!=0,'Cholesterol'].mean()
print(ch_mean)

df['Cholesterol'] = df['Cholesterol'].replace(0,ch_mean)
df['Cholesterol'] = df['Cholesterol'].round(1)
print(df['Cholesterol'])

'''resting bp cleaning'''
RestingBP_mean = df.loc[df['RestingBP']!=0,'RestingBP'].mean()
print(RestingBP_mean)

df['RestingBP'] = df['RestingBP'].replace(0,RestingBP_mean)
df['RestingBP'] = df['RestingBP'].round(1)
print(df['RestingBP'])


sns.countplot(x = df['Sex'],hue = df['HeartDisease'])               # connection of heart disease with sex
plt.show()
sns.countplot(x = df['ChestPainType'],hue = df['HeartDisease'])     # connection of heart disease with chest pain type
plt.show()
sns.countplot(x = df['FastingBS'],hue = df['HeartDisease'])     # connection of heart disease with chest pain type
plt.show()
sns.boxplot(x='HeartDisease',y='Cholesterol',data = df)
plt.show()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

## Data Preprocessing and Cleaning
df_encode = pd.get_dummies(df)
df_encode = df_encode.astype(int)

print(df_encode)

# Feature Engineering and Extraction (Selection)

df_encode['Age_group'] = pd.cut(
    df_encode['Age'],
    bins = [0,40,50,60,100],
    labels = ['Young','Middle','Senior','Elderly']
)

df_encode["BPCategory"] = pd.cut(
    df_encode["RestingBP"],
    bins=[0, 120, 140, 180, float('inf')],
    labels=["Normal", "Elevated", "High", "Very High"]
)

df_encode["CholesterolCategory"] = pd.cut(
    df_encode["Cholesterol"],
    bins=[0, 200, 240, float('inf')],
    labels=["Normal", "Borderline", "High"]
)

df_encode["MaxHRCategory"] = pd.cut(
    df_encode["MaxHR"],
    bins=[0, 100, 140, 180, float('inf')],
    labels=["Low", "Moderate", "High", "Very High"]
)

df_encode = pd.get_dummies(
    df_encode,
    columns=[
        "Age_group",
        "BPCategory",
        "CholesterolCategory",
        "MaxHRCategory"
    ],
    dtype=int
)

from scipy.stats import pearsonr
import pandas as pd

# List of features to check against target
selected_features = [
    'Age',
    'RestingBP',
    'Cholesterol',
    'FastingBS',
    'MaxHR',
    'Oldpeak',
    'Sex_F',
    'Sex_M',
    'ChestPainType_ASY',
    'ChestPainType_ATA',
    'ChestPainType_NAP',
    'ChestPainType_TA',
    'RestingECG_LVH',
    'RestingECG_Normal',
    'RestingECG_ST',
    'ExerciseAngina_N',
    'ExerciseAngina_Y',
    'ST_Slope_Down',
    'ST_Slope_Flat',
    'ST_Slope_Up',
    'Age_group_Young',
    'Age_group_Middle',
    'Age_group_Senior',
    'Age_group_Elderly',
    'BPCategory_Normal',
    'BPCategory_Elevated',
    'BPCategory_High',
    'BPCategory_Very High',
    'CholesterolCategory_Normal',
    'CholesterolCategory_Borderline',
    'CholesterolCategory_High',
    'MaxHRCategory_Low',
    'MaxHRCategory_Moderate',
    'MaxHRCategory_High',
    'MaxHRCategory_Very High'
]

correlations = {
    feature: pearsonr(df_encode[feature], df_encode['HeartDisease'])[0]
    for feature in selected_features
}

correlation_df = pd.DataFrame(
    list(correlations.items()),
    columns=['Feature', 'Pearson Correlation']
)

print(correlation_df.sort_values(by='Pearson Correlation', ascending=False))

# scaling
from sklearn.preprocessing import StandardScaler
cols=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
scaler = StandardScaler()
df_encode[cols]=scaler.fit_transform(df_encode[cols])
print(df_encode.head())

final_df = df_encode[['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak','Sex_M','ChestPainType_ASY','ChestPainType_ATA','ChestPainType_NAP','RestingECG_LVH','RestingECG_ST','ExerciseAngina_Y','ST_Slope_Flat','ST_Slope_Up','Age_group_Middle','Age_group_Senior','Age_group_Elderly','BPCategory_Elevated','BPCategory_High','BPCategory_Very High','CholesterolCategory_Borderline','CholesterolCategory_High','MaxHRCategory_Moderate','MaxHRCategory_High','MaxHRCategory_Very High','HeartDisease']]

