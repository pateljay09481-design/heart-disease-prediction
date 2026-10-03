import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn. linear_model import LogisticRegression
from sklearn. naive_bayes import GaussianNB
from sklearn. tree import DecisionTreeClassifier
from sklearn. svm import SVC
from sklearn. neighbors import KNeighborsClassifier
import joblib


df=pd.read_csv('heart.csv')
print(df)
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
numerical_columns=['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']

for col in numerical_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col],kde=True,bins=20)
    plt.show()

print(df['Cholesterol'].value_counts())
ch_mean=df.loc[df['Cholesterol']!=0,'Cholesterol'].mean()
df['Cholesterol']=df['Cholesterol'].replace(0,ch_mean)
df['Cholesterol']=df['Cholesterol'].round(2)

ch_meanr=df.loc[df['RestingBP']!=0,'RestingBP'].mean()
df['RestingBP']=df['RestingBP'].replace(0,ch_meanr)
df['RestingBP']=df['RestingBP'].round(2)

print(df.columns)
categoriacal_columns=[ 'Sex', 'ChestPainType', 'RestingECG',  'ExerciseAngina',  'ST_Slope']

for col in categoriacal_columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[col],hue=df['HeartDisease'])
    plt.show()

for col in numerical_columns:
   plt.figure(figsize=(6,4))
   sns.boxplot(y=df[col],x=df['HeartDisease'])
   plt.show()
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

df_clean=df.copy()


df_clean=pd.get_dummies(df_clean,drop_first=True)
df_clean=df_clean.astype(int)
print(df_clean.head())
print(df.columns)
numeric_col=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
scaler=StandardScaler()
df_clean[numeric_col]=scaler.fit_transform(df_clean[numeric_col])
print(df_clean.head())



x=df_clean.drop('HeartDisease',axis=1)
y=df_clean['HeartDisease']


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(X_train)
x_test_scaled=scaler.fit_transform(X_test)

models={
    "logistic regreesion": LogisticRegression(),
    "knn":  KNeighborsClassifier(),
    "naive byes":GaussianNB(),
    "decision tree":DecisionTreeClassifier(),
    "svm":SVC()
}
result=[]



for name,model in models.items () :
   model.fit(x_train_scaled , y_train)
   y_pred = model.predict(x_test_scaled)
   acc = accuracy_score(y_test, y_pred)
   f1 = f1_score(y_test,y_pred)
   result.append({
      'model' : name,
      'Accuracy': round(acc,4),
      "f1 score": round(f1,4)
      })
   
   
print(result)

joblib.dump(models['knn'],'knn_heart.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(x.columns.to_list(),'columns.pkl')


       
      
