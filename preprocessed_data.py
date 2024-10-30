# inport the libaries that are going to be used:
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
import numpy as np
from sklearn.impute import SimpleImputer
import category_encoders as ce

#-----------------------------------------------------------------------------------------

# Data Wrangling
def fillnulls(data):
    # filling null values with median, mode:
    data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
    data['MasVnrType'] = data['MasVnrType'].fillna(data['MasVnrType'].mode()[0])
    data['BsmtExposure'] = data['BsmtExposure'].fillna(data['BsmtExposure'].mode()[0])
    data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
    data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].median())
    # filling null values with zero:
    data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
    # filling selected columns with np.nan (or 'None' if you want a string):
    fill_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
                 'BsmtQual', 'BsmtCond', 'BsmtFinType2', 'BsmtFinType1', 'FireplaceQu', 'MiscFeature']
    for column in fill_cols:
        data[column] = data[column].fillna(np.nan)  # or 'None' for strings
    return data


# ----------------------------------------------------------------------------------------------------------------------------------

# feature engineering:
def ft_engineering(data):
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    data['TotalBathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))
    data['TotalPorchSF'] = (data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch'] + data['WoodDeckSF'])
    data['Age'] = data['YrSold'] - data['YearBuilt']
    data['AgeRemod'] = data['YrSold'] - data['YearRemodAdd']
    data['TotalQual_Cond'] = data['OverallQual'] + data['OverallCond']
    data = data.drop(columns=['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath', 'BsmtFullBath',
                      'BsmtHalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch',
                      'WoodDeckSF', 'YrSold', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond'])
    return data

#-----------------------------------------------------------------------------------------------------------------------------------------

# encoding: ordinal
def ordinal_encode(data):
    # Define ordinal features
    ordinal_features = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                        'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 
                        'GarageQual', 'GarageCond']
    # Define the mapping dictionary to handle NaN values, and the encodding to ordinal:
    mapping_dict = {
        'LotShape': {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1, np.nan: 0},
        'LandSlope': {'Gtl': 3, 'Mod': 2, 'Sev': 1, np.nan: 0},
        'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0},
        'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0},
        'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'None': 1, np.nan: 0},
        'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'None': 1, np.nan: 0},
        'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, np.nan: 0},
        'BsmtFinType1': {'GLQ': 7, 'ALQ': 6, 'BLQ': 5, 'Rec': 4, 'LwQ': 3, 'Unf': 2, 'None': 1, np.nan: 0},
        'BsmtFinType2': {'GLQ': 7, 'ALQ': 6, 'BLQ': 5, 'Rec': 4, 'LwQ': 3, 'Unf': 2, 'None': 1, np.nan: 0},
        'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0},
        'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, np.nan: 0},
        'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'None': 1, np.nan: 0},
        'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'None': 1, np.nan: 0},
        'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'None': 1, np.nan: 0},
    }
    # Apply the mapping dictionary to each feature by looping:
    for feature in ordinal_features:
        data[feature] = data[feature].map(mapping_dict[feature])
    return data

# imputing the features from ordinal_encode:
    # This function choose the imputation strategy based on skewness
def choose_imputation_strategy(column_name, skew_value):
    if np.abs(skew_value) < 0.5:  # Less skewed, use mean
        return 'mean'
    else:  # More skewed, use median since it's not a normal distribution 
        return 'median'
    
# for train:
def impute_train(data, columns, imputer_dict):
    # Dictionary to store imputers
    imputer_dict = {}
    # Check skewness of each numeric column in the training data
    skewed_columns = data[columns].skew()
    # Fit imputers based on the skewness in the training data
    for column in columns:
        strategy = choose_imputation_strategy(column, skewed_columns[column]) # mean or median
        imputer = SimpleImputer(strategy=strategy)
        data[column] = imputer.fit_transform(data[[column]])
        imputer_dict[column] = imputer
    return data, imputer_dict

# For test: 
# Impute test data using pre-fitted imputers from training
def impute_test(data, columns, imputer_dict):
    # Transform the test data using the imputers fitted on training data
    for column in columns:
        data[column] = imputer_dict[column].transform(data[[column]]) # only transforming the test
    return data 



# encode: Target 
# For train:
def target_encode_train(data, target_cols, target):
    # Dictionary to store the fitted encoders
    targ_encoders = {}
    # Iterate over the columns to encode
    for col in target_cols:  
        target_encoder = ce.TargetEncoder(cols=[col])
        # Fit the encoder on the training data and transform it
        data[col] = target_encoder.fit_transform(data[col], data[target])
        # Store the fitted encoder for future use on the test set
        targ_encoders[col] = target_encoder
    return data, targ_encoders

# For test: 
def target_encode_test(data, targ_encoders, target_cols):
    # Iterate over the columns that were encoded in the training set
    for col in target_cols:
        target_encoder = targ_encoders[col]
        # transform the test data:
        data[col] = target_encoder.transform(data[col])
    return data



# encode: OneHot
# For train:
def one_hot_encode_train(data, one_hot_cols):
    # Dictionary to store the fitted encoders
    onehot_encoders = {}
    # Iterate over the columns to encode
    for col in one_hot_cols:
        one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        # Fit and transform the training data
        encoded_feature = one_hot_encoder.fit_transform(data[[col]])
        # Store the encoder for future use on the test set
        onehot_encoders[col] = one_hot_encoder
        # Convert the encoded features to a DataFrame with proper column names
        encoded_df = pd.DataFrame(encoded_feature, columns=one_hot_encoder.get_feature_names_out([col]), index=data.index)
        # Concatenate the encoded DataFrame with the original data
        data = pd.concat([data, encoded_df], axis=1)
        # Drop the original column
        data.drop(columns=[col], inplace=True)
    return data, onehot_encoders

# for test:
# Function to transform the test set using the fitted encoders
def one_hot_encode_test(data, onehot_encoders, one_hot_cols):
    # Iterate over the columns to encode th features from the training set
    for col in one_hot_cols:
        one_hot_encoder = onehot_encoders[col]
        # Transform the test data using the fitted encoder
        encoded_feature = one_hot_encoder.transform(data[[col]])
        # Convert the encoded features to a DataFrame with proper column names
        encoded_df = pd.DataFrame(encoded_feature, columns=one_hot_encoder.get_feature_names_out([col]), index=data.index)
        # Concatenate the encoded DataFrame with the original test data
        data = pd.concat([data, encoded_df], axis=1)
        # Drop the original column
        data.drop(columns=[col], inplace=True)
    return data, onehot_encoders

#------------------------------------------------------------------------------------------------------------------------

# 
