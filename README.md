# House-Prices---Regression-Techniques

## Summary ( your results and a discussion of the project - what you did to the data, which models you used, and how well you think the models performed.): 
The goal of this project was to predict house sale prices using machine learning techniques in Python. After preparing the data to reduce dimensionality, I tested models such as Linear Regression and Random Forest, while making several submissions (not included in the notebook). Based on the Root Mean Squared Error (RMSE), I found that XGBoost was the most efficient model due to its ability to handle more parameters effectively. Key features used in the model included 'TotalSF', 'LotArea', and 'GrLivArea'. In conclusion, XGBoost provided the best balance between efficiency and accuracy for forecasting house prices.

## DataSet info:
The datasets used in this project can be accessed from this repository or through the following link:
<a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data" target="_blank"> Here are the DataSets</a>
- train.csv
- test.csv
- data_description.txt - Detailed feature descriptions for both training and test sets.
- submission.csv - Contains predicted SalesPrice values along with their corresponding IDs.
The training dataset contains 81 features, while the test dataset contains 80 features. The SalesPrice feature is available in the training set but not in the test set.

## Techniques used:
- Data preparation before modeling: Handling missing values, encoding categorical variables, and feature scaling to ensure the dataset was ready for modeling.
- Visualizing data distributions with histograms: Used to assess feature skewness for imputation and to better understand the data.
- XGBoost: Employed the DMatrix to improve training speed and memory efficiency.

## Features Used:
I dropped several features from both the training and testing datasets, including 'Alley', 'PoolQC', 'Fence', and 'Utilities', due to their high percentage of missing values. After feature engineering, I also removed 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'OverallQual', and 'OverallCond'. These were replaced by features like 'TotalQual_Cond', 'AgeRemod', 'Age', 'TotalPorchSF', 'TotalBathrooms', and 'TotalSF' to improve model performance. Combining these features reduced the dimensionality of the dataset, contributing to a more efficient model and helping to prevent overfitting. The remaining features were left unchanged, as they were critical in improving the model's accuracy. The top 10 important features identified by the XGBoost model were: 'Id', 'LotArea', 'LotFrontage', 'BsmtUnfSF', 'GrLivArea', 'MSSubClass', 'TotalPorchSF', 'GarageArea', 'BsmtFinSF1', and 'TotalSF'.

## Evaluation:
The model's performance was evaluated using the Root Mean Squared Error (RMSE) between the logarithms of the predicted values and the observed sales prices.

## Results and conclusion:
The images below display the model's results. The first image shows the predicted SalePrice values alongside their respective IDs, while the second image presents the RMSE score on Kaggle.
<img width="123" alt="Screenshot 2024-09-25 at 3 32 05 PM" src="https://github.com/user-attachments/assets/04ee782f-b97b-4051-a3b8-08d14ce0bca8">
<img width="737" alt="Screenshot 2024-09-25 at 3 30 46 PM" src="https://github.com/user-attachments/assets/067afa08-95f1-407f-9da8-7e3166d576ad">
Initially, I used linear regression, but the RMSE results showed it wasn’t the best choice. I then tested the Random Forest model, which was also less accurate. Ultimately, I found that XGBoost provided the most accurate and efficient predictions.
