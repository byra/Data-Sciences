#Data analysis and wrangling
import pandas as pd

#Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Machine learning
from sklearn.linear_model import LogisticRegression

#Performance computation 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#Load data in program
train_dataset = pd.read_csv('/Users/Byra/Documents/Data-Sciences/python-projects/TitanicSurvivalPredictions/Dataset/train.csv', index_col='PassengerId')
test_dataset = pd.read_csv('/Users/Byra/Documents/Data-Sciences/python-projects/TitanicSurvivalPredictions/Dataset/test.csv', index_col='PassengerId')
actual_prediction = pd.read_csv('/Users/Byra/Documents/Data-Sciences/python-projects/TitanicSurvivalPredictions/Dataset/gender_submission.csv', index_col='PassengerId').Survived

#Observe datasets features for type and missing variables
train_dataset.index
train_dataset.columns
train_dataset.info()
train_dataset.describe()
train_dataset.describe(include=['O'])
train_dataset.isnull().any()

test_dataset.index
test_dataset.columns
test_dataset.info()
test_dataset.describe()
test_dataset.describe(include=['O'])
test_dataset.isnull().any()

#Seperate Target and Features for train dataset
train_target_variable = train_dataset.Survived
train_feature_variables = train_dataset.drop(labels='Survived', axis=1)

#Dropping the unwanted or features with no effect in both train and test data sets
train_feature_variables = train_feature_variables.drop(['Name','Ticket', 'Cabin'], axis=1)
test_dataset = test_dataset.drop(['Name','Ticket', 'Cabin'], axis=1)

#Combining train and test datasets
combine = [train_feature_variables, test_dataset]

#Filling missing values in both train and test 
for dataset in combine:
    frequent_Embarked = dataset.Embarked.mode()[0]
    dataset.Embarked = dataset.Embarked.fillna(frequent_Embarked)
    mean_Age = dataset.Age.mean()
    dataset.Age = dataset.Age.fillna(mean_Age)
    frequent_Fare = dataset.Fare.mode()[0]
    dataset.Fare = dataset.Fare.fillna(frequent_Fare)
#Changing Categorical values
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#Initiate SVC
logisticRegression = LogisticRegression()

#Fit the model
logisticRegression.fit(train_feature_variables, train_target_variable)

#Predict the test target
test_prediction =  logisticRegression.predict(test_dataset)

#Performance of model
print("Accuracy of the Model\n{}".format(accuracy_score(actual_prediction, test_prediction)))
print("Confusion Matrix:\n{}".format(confusion_matrix(actual_prediction, test_prediction)))
print(classification_report(actual_prediction, test_prediction))

