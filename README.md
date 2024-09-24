# House-Prices---Advanced-Regression-Techniques

## Goal and Purpose:
Using Python the overall goal is to predict the sales price for each house. The column Id is used in the test set to predict the values in SalePrice. 

## DataSet info:
The datasets can be found in this link or can be found in files in this repository.
<a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data" target="_blank"> Here are the DataSets</a>
- train.csv 
- test.csv 
- data_description.txt - full description of each feature that is in the training and the testing set. 
- submission.csv - the predicted values of the SalesPrice along with their Id.
For the train data set there is a total of 81 features while in the test set, there is a total of 80 features. The training set has the SalesPrice feature while the test set doesn't. 

## Techniques used:
- Using histograms to see the distribution of the data as well as the skew for the features for imputation purposese. 
- XGBOOST: using the Dmatrix so for a faster training speed and its efficency for memory. 

## Evaluation:
The submission is evaluated by using Root-Means-Squared-Error between the logarithmic of the predicted values and the logarithmic values observed in the sales price. 

