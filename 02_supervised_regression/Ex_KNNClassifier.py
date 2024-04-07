# AI Singapore
# Regression 2 Exercise
# Exercise: Building a Regression job template

# 1. Import required libraries
import numpy as np 
import pandas as pd 
import datetime as d

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_squared_error, r2_score

import joblib
import os

# Information on Data
# https://www.kaggle.com/c/home-data-for-ml-course/data

# Custom Classes and Functions
def display_df_info(df_name, my_df, v=False):
    """Convenience function to display information about a dataframe"""

    print("Data: {}".format(df_name))
    print("Shape (rows, cols) = {}".format(my_df.shape))
    print("First few rows...")
    print(my_df.head())

    # Optional: Display other optional information with the (v)erbose flag
    if v:
        print("Dataframe Info:")
        print(my_df.info())

class GetAge(BaseEstimator, TransformerMixin):
    """Custom Transformer: Calculate age (years only) relative to current year. Note that 
    the col values will be replaced but the original col name remains. When the transformer is 
    used in a pipeline, this is not an issue as the names are not used. However, if the data 
    from the pipeline is to be converted back to a DataFrame, then the col name change should 
    be done to reflect the correct data content."""

    def fit(self, X, y=None):
        return self
    
    def transform(self,X):
        current_year = int(d.datetime.now().year)
        X['YearBuilt'] = current_year - X['YearBuilt']
        return X

def main():
    
    # DATA INPUT
    ############
    file_path = f"{os.path.dirname(os.path.realpath(__file__))}/data/train.csv"
    input_data = pd.read_csv(file_path, index_col='Id')
    display_df_info("Raw Input", input_data)

    # Seperate out the outcome variable from the loaded dataframe
    output_var_name = 'SalePrice'
    output_var = input_data[output_var_name]
    input_data.drop(output_var_name, axis=1, inplace=True)

    # DATA ENGINEERING / MODEL DEFINITION
    #####################################

    # Subsetting the columns: define features to keep
    feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'HouseStyle']
    features = input_data[feature_names]
    display_df_info('Features before Transform', features, v=True)

    # Create the pipeline ...
    # 1. Pre-processing
    # Define variables made up of lists. Each list is a set of columns that will go through the same data transformations.
    numerical_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    categorical_features = ['HouseStyle']
    
    """TASK:
    Define the data processing steps (transformers) to be applied to the numerical features in the dataset.

    At a minimum, use 2 transformers: GetAge() and one other. Combine them using make_pipeline() or Pipeline()
    """
    preprocess = make_column_transformer(
        (make_pipeline(GetAge(), SimpleImputer(), StandardScaler()), numerical_features),
        (OneHotEncoder(), categorical_features)
    )
    
    # 2. Combine pre-processing with ML algorithm
    pipeline = make_pipeline(
        preprocess,        
        KNeighborsRegressor())   
    params = {'kneighborsregressor__n_neighbors' : range(2,21), 'kneighborsregressor__weights' : ['uniform','distance']} 
    model = GridSearchCV(pipeline, params, cv=5, scoring='neg_mean_squared_error')

    # TRAINING
    ##########
    # Train/Test Split
    """TASK:
    Split the data in test and train sets by completing the train_test_split function below. Define a random_state value so that 
    the experiment is repeatable.
    """
    x_train, x_test, y_train, y_test = train_test_split(features, output_var, test_size=0.3, random_state=42)
    # Train the pipeline
    model.fit(x_train, y_train)

    # Optional: Train with cross-validation and/or parameter grid search

    # SCORING/EVALUATION
    ####################
    # Fit the model on the test data
    pred_test = model.predict(x_test)

    print("Best Parameters chosen : {}".format(model.best_params_))
    
    # Display the results of the metrics
    """TASK:
    Calculate the RMSE and Coeff of Determination between the actual and predicted sale prices. 
        
    Name your variables rmse and r2 respectively.
    """
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2 = r2_score(y_test, pred_test)
    print("Results on Test Data")
    print("####################")
    print("RMSE: {:.2f}".format(rmse))
    print("R2 Score: {:.5f}".format(r2))
    
    # Compare actual vs predicted values
    """TASK:
    Create a new dataframe which combines the actual and predicted Sale Prices from the test dataset. You
    may also add columns with other information such as difference, abs diff, %tage difference etc.
    
    Name your variable compare
    """
    compare = pd.DataFrame({'Actual': y_test, 'Predicted': pred_test, 'Difference': round(y_test - pred_test, 2)})
    display_df_info('Actual vs Predicted Comparison', compare)

    # Save the model 
    with open('my_model_lr.joblib', 'wb') as fo:  
        joblib.dump(model, fo)


if __name__ == '__main__':
    main()
