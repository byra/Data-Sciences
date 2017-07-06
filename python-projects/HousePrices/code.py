#Data analysis
import pandas as pd
from sklearn.linear_model import ElasticNetCV
import numpy as np

#Load the dataset
train_dataset = pd.read_csv('Dataset/train.csv', index_col='Id')
test_dataset = pd.read_csv('Dataset/test.csv', index_col='Id')
actual_prediction = pd.read_csv('Dataset/sample_submission.csv', index_col='Id').SalePrice

#Separting feature variables and target variables
train_feature_variables = train_dataset.drop(labels='SalePrice', axis=1)
train_target_variable = train_dataset.SalePrice

#View data
train_feature_variables.shape
train_feature_variables.index
train_feature_variables.columns
train_feature_variables.info()
train_feature_variables.describe()
train_feature_variables.describe(include=['O'])
train_feature_variables.isnull().any()

test_dataset.shape
test_dataset.index
test_dataset.columns
test_dataset.info()
test_dataset.describe()
test_dataset.describe(include=['O'])
test_dataset.isnull().any()

#Filling missing values 
train_feature_variables = train_feature_variables.fillna(train_feature_variables.mean())
train_feature_variables = train_feature_variables.apply(lambda x:x.fillna(x.value_counts().index[0]))

test_dataset = test_dataset.fillna(test_dataset.mean())
test_dataset = test_dataset.apply(lambda x:x.fillna(x.value_counts().index[0]))


#Converting categorical values to numeric
train_feature_variablesCatColumns = train_feature_variables.select_dtypes(['object']).columns
train_feature_variables[train_feature_variablesCatColumns] = train_feature_variables[train_feature_variablesCatColumns].apply(lambda x: x.astype('category'))
train_feature_variables[train_feature_variablesCatColumns] = train_feature_variables[train_feature_variablesCatColumns].apply(lambda x: x.cat.codes)

test_datasetCatColumns = test_dataset.select_dtypes(['object']).columns
test_dataset[test_datasetCatColumns] = test_dataset[test_datasetCatColumns].apply(lambda x: x.astype('category'))
test_dataset[test_datasetCatColumns] = test_dataset[test_datasetCatColumns].apply(lambda x: x.cat.codes)


#Identifying outliers and removing them
q25, q75 = np.percentile(train_target_variable, [25,75])
iqr = q75 - q25    
min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)
train_feature_variables = train_feature_variables[(train_target_variable > min) & (train_target_variable < max)]
train_target_variable = train_target_variable[(train_target_variable > min) & (train_target_variable < max)]



#Initiate model object
elasticNet = ElasticNetCV(l1_ratio = [ 0.9, 0.95, 1],alphas = [1, 3, 6], max_iter = 50000, cv = 10)
elasticNet.fit(train_feature_variables, train_target_variable)
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("Try again for more precision with l1_ratio centered around " + str(ratio))
elasticNet = ElasticNetCV(l1_ratio = [ratio * .95,  ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],alphas = [ 1, 3, 6, 10], max_iter = 50000, cv = 10)
#Fit model
elasticNet.fit(train_feature_variables, train_target_variable)

#Predict model
test_prediction = elasticNet.predict(test_dataset)

#Compute rmse
print("RMSE: {}".format(np.sqrt(((actual_prediction - test_prediction)**2).mean())))