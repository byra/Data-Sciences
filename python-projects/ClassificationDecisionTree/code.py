#import required packages
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

#load data_1 in program
train_dataset_1 = pd.read_csv('/Users/Byra/Desktop/dataset/training_set1.csv')
validation_dataset_1 = pd.read_csv('/Users/Byra/Desktop/dataset/validation_set1.csv')
test_dataset_1 = pd.read_csv('/Users/Byra/Desktop/dataset/test_set1.csv')

#append train and validation dataset 
dataset_1 = train_dataset_1.append(validation_dataset_1, ignore_index= True)

#Seperate Target and Features for train dataset 
train_target_variable_1 = dataset_1.Class
train_feature_variables_1  = dataset_1.drop(labels='Class', axis=1)

actual_prediction_1 = test_dataset_1.Class
test_feature_variables_1 = test_dataset_1.drop(labels='Class', axis=1)

#Initiate DecisionTreeClassifier
decisionTreeClassifer = DecisionTreeClassifier()

#Fit the model
decisionTreeClassifer.fit(train_feature_variables_1, train_target_variable_1)

#Predict the test target
test_prediction_1 =  decisionTreeClassifer.predict(test_feature_variables_1)

#Delete Object
del decisionTreeClassifer

#Generate results
print("Confusion Matrix for dataset 1:\n{}".format(confusion_matrix(actual_prediction_1, test_prediction_1)))
print("Accuracy of the Model 1\n{}".format(accuracy_score(actual_prediction_1, test_prediction_1)))
print(classification_report(actual_prediction_1, test_prediction_1))

#load data_2 in program
train_dataset_2 = pd.read_csv('/Users/Byra/Desktop/dataset/training_set2.csv')
validation_dataset_2 = pd.read_csv('/Users/Byra/Desktop/dataset/validation_set2.csv')
test_dataset_2 = pd.read_csv('/Users/Byra/Desktop/dataset/test_set2.csv')

#append train and validation dataset 
dataset_2 = train_dataset_1.append(validation_dataset_1, ignore_index= True)

#Seperate Target and Features for train dataset 
train_target_variable_2 = dataset_2.Class
train_feature_variables_2  = dataset_2.drop(labels='Class', axis=1)

actual_prediction_2 = test_dataset_2.Class
test_feature_variables_2 = test_dataset_2.drop(labels='Class', axis=1)

#Initiate DecisionTreeClassifier
decisionTreeClassifer = DecisionTreeClassifier()

#Fit the model
decisionTreeClassifer.fit(train_feature_variables_2, train_target_variable_2)

#Predict the test target
test_prediction_2 =  decisionTreeClassifer.predict(test_feature_variables_2)

#Generate results
print("Confusion Matrix  for dataset 2:\n{}".format(confusion_matrix(actual_prediction_2, test_prediction_2)))
print("Accuracy of the Model 2\n{}".format(accuracy_score(actual_prediction_2, test_prediction_2)))
print(classification_report(actual_prediction_2, test_prediction_2))