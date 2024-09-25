# House-Prices---Regression-Techniques

## Goal:
The main objective is to forecast each house's sales price using Python. In the test set, the values in SalePrice are indicated by the column Id. 

## Summary ( your results and a discussion of the project - what you did to the data, which models you used, and how well you think the models performed.): 


## DataSet info:
The datasets are available as files in this repository or via this URL.
<a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data" target="_blank"> Here are the DataSets</a>
- train.csv
- test.csv
- data_description.txt - A full description of each feature in both the training and testing sets.
- submission.csv - Contains the predicted SalesPrice values along with their corresponding IDs.
The training dataset contains 81 features, while the test dataset contains 80 features. The SalesPrice feature is present in the training set but absent from the test set.

## Techniques used:
- Using histograms to visualize the data distribution and assess the skew of the features for imputation purposes.
- XGBOOST: employing the Dmatrix to increase training speed and memory efficiency.

## Features used and why:


## Evaluation:
The submission is evaluated using the Root Mean Squared Error between the predicted values' logarithms and the sales prices' observed logarithms.

## Results (images of your EDA or your results):

<img width="737" alt="Screenshot 2024-09-25 at 3 30 46 PM" src="https://github.com/user-attachments/assets/067afa08-95f1-407f-9da8-7e3166d576ad">




You may also choose to discuss any differences in feature importance from one of the linear regression models to one of the tree-based models.
