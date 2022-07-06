## Classification of survival of a passenger in Titanic dataset using TensorFlow

### Dataset used
The Titanic dataset is loaded as a Pandas DataFrame by Seaborn. 
Reference for Seaborn datasets:
https://seaborn.pydata.org/generated/seaborn.load_dataset.html

### Test condition
The classifier used is multi-layer perceptron (MLP) in TensorFlow. 
The following features of Titanic dataset are used in classification: pclass, sex, 
age, sibsp, parch, fare.
The class label is "survived". 
Numerical values of the following features 'pclass', 'age', 'sibsp', 'parch', 'fare' 
are scaled to values between 0 and 1 by using MinMaxScaler of scikit-learn. The 
MinMAxScaler is used to fit and transform the training set only. MinMaxScaler is 
used to transform the test set. 25% of data is used as test set and 75% as 
training set.  
A multi-layer perceptron with 2 hidden layers in TensorFlow is used in 
classification.
The result of MLP from TensorFlow is compared with the result of decision tree classifier in
scikit-learn. The default settings are used for the decision tree classifer.
The experiment is repeated for 10 times and the average accuracy for MLP and 
decision tree classifier is found. 10 different partitions of data to training set
and test set are used. 

### Test result
Average accuracy for MLP = 81.51%\
Average accuracy for decision tree classifier = 75.92%
