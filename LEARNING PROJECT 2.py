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

from sklearn.preprocessing import StandardScaler
cols=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
scaler = StandardScaler()
df_encode[cols]=scaler.fit_transform(df_encode[cols])
print(df_encode.head())































