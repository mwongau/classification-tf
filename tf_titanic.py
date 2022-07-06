import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

# Titanic dataset is loaded by Seaborn.
# reference: https://seaborn.pydata.org/generated/seaborn.load_dataset.html 
titanic=sns.load_dataset('titanic')
print(titanic.info())
print(titanic.head())
df = titanic.copy()
encoder = LabelEncoder()
df.iloc[:,2]= encoder.fit_transform(df.iloc[:,2].values) # Encode sex features to 1 & 0
data = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
data = data.dropna() # remove rows with missing values
print('\nlength of dataset after pre-processing', len(data))
print(data.info())
print(data.head())
labels = data.pop('survived')

# ensure all data are floating point values
data = data.astype('float32')

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)

scaler = MinMaxScaler()
X_train[['pclass', 'age', 'sibsp', 'parch', 'fare']] = scaler.fit_transform(X_train[['pclass', 'age', 'sibsp', 'parch', 'fare']])
X_test[['pclass', 'age', 'sibsp', 'parch', 'fare']] = scaler.transform(X_test[['pclass', 'age', 'sibsp', 'parch', 'fare']])
print('X_train information')
print(X_train.info())
print(X_train.head())
print('X_test information')
print(X_test.info())
print(X_test.head())

X_train = np.array(X_train)
X_test = np.array(X_test)

# determine the number of input features
n_features = X_train.shape[1]

# define model
model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(n_features,)))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(X_train, y_train, epochs=300, batch_size=32, verbose=2)

# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print('TensorFlow: test Accuracy: %.4f' % acc)

cls_tree = DecisionTreeClassifier()
cls_tree = cls_tree.fit(X_train, y_train)
score = cls_tree.score(X_test, y_test)
print('Decision tree: test accuracy: %.4f' % score)

fp = open('res.txt', 'a')
fp.write(str(acc))
fp.write('\n')
fp.write(str(score))
fp.write('\n')
fp.close()



