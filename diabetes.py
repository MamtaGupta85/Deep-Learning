# -*- coding: utf-8 -*-
"""diabetes.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GTjnIaw4lKqVROD_flIFoGcX0q97V45N
"""

#@title Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Deep learning/diabetes.csv')

#dimensions of the dataset
df.shape

#first five rows of the dataset
df.head()

#check for the null values and dataytypes
df.info()

#@title Exploratory Data Analysis and Preprocessing

#checking for the nan values
df.isna().sum()

df.describe().T

# The distribution of the Outcome variable was examined.
df.groupby('class')['class'].value_counts()*100/len(df)

# The classes of the outcome variable were examined.
df.groupby('class')['class'].value_counts()

#mapping the class Negative:0 and Positive:1 for the output variable.
df['class']=df['class'].map({'Negative':0, 'Positive':1})

df.groupby('Gender')['Gender'].value_counts()

#mapping the class Male:0 and Female:1 for the Gender variable.
df['Gender']=df['Gender'].map({'Male':0, 'Female':1})

df.groupby('Polyuria')['Polyuria'].value_counts()

df.groupby('visual blurring')['visual blurring'].value_counts()

#@title Meta Data: Map the 'Yes': 1 and 'No': 0
p=[]
for col in df.columns:
  for row in df[col]:
    if row=='Yes' or row=='No':
      p.append(col)
      break
    else:
      continue

for i in df[p]:
  df[i]= df[i].map({"Yes":1, "No":0})

df.head()

df.info()

#@title Univariate Analysis

def dist_plot(x):
  plt.rcParams['figure.figsize']=(5,10)
  sns.displot(x=x, data=df,color='firebrick',bins=50)
  plt.title(f'Countplot of {x}')
  plt.xlabel(x)
  plt.tight_layout()

dist_plot('Age')

dist_plot('Gender')

plt.rcParams['figure.figsize']=(8,5)
sns.countplot(y=df['class'],hue= df['class'],palette='summer')
plt.title('Countplot of classes')
plt.xlabel('Class')
plt.tight_layout()

def count_xplot(x,fig):
  plt.rcParams['figure.figsize']=(20,30)
  plt.subplot(6,2,fig)
  sns.countplot(y=df[x], hue=df[x], palette='magma')
  plt.title(f'Countplot of {x}')
  plt.xlabel(x)

count_xplot('Polyuria',1)
count_xplot('Polydipsia',2)
count_xplot('sudden weight loss',3)
count_xplot('weakness',4)
count_xplot('Polyphagia',5)
count_xplot('Genital thrush',6)

@title Bivariate Analysis
plt.rcParams['figure.figsize']=(5,5)
sns.stripplot(x=df['class'],y=df['Age'],hue=df['class'],palette='cool')

sns.lineplot(x=df['class'], y=df['Age'],color='brown')
plt.title('Relationship between Age and class')

sns.boxplot(x=df['class'],y=df['Age'],color='firebrick')

#Correlation Analysis
plt.rcParams['figure.figsize']=(10,5)
sns.heatmap(df.corr(), annot=True)

#Multicollinearity Analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor

x= df
VIF=pd.DataFrame()
VIF['Factors']= x.columns
VIF['VIF']= [variance_inflation_factor(x.values,i) for i in range(len(x.columns))]

VIF

#separating the input and output dataset
input = df.drop(columns=['class'])
output = df.iloc[:,-1]
print(f'the dimensions of the input dataset is {input.shape}')
print(f'the dimensions of the output dataset is {output.shape}')

#splitting the data for training and testing
train_x,test_x,train_y,test_y= train_test_split(input,output,test_size=0.2)

print(f'the dimensions of the training dataset is {train_x.shape}')
print(f'the dimensions of the testing dataset is {test_y.shape}')

#Scaling the training and testing sets

sc= StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)

#@title Deep Learning Model Formation

model = Sequential()

model.add(Dense(10, activation='relu', input_dim=16))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

#Compiling the model
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=1)

history = model.fit(train_x,train_y,epochs=100,validation_split=0.2, callbacks=callback)

plt.rcParams['figure.figsize']=(8,4)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss','val_loss'], loc='upper right')
plt.xlabel('No of epochs')
plt.ylabel('loss')
plt.title('Actual loss and validation loss')
plt.tight_layout()

plt.rcParams['figure.figsize']=(8,4)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy','val_accuracy'], loc='lower right')
plt.xlabel('No of epochs')
plt.ylabel('Accuracy')
plt.title('Total accuracy and validation accuracy')
plt.tight_layout()

y_log = model.predict(test_x)

y_pred = np.where(y_log>0.5,1,0)

accuracy_score(test_y, y_pred)

#evaluating the model on manual basis

print(f' the predicted class for the row 49 is {y_pred[49]} and the actual class is {test_y.iloc[49]}')

print(f' the predicted class for the row 68 is {y_pred[68]} and the actual class is {test_y.iloc[68]}')