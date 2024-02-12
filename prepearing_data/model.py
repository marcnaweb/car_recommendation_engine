import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle
import os
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

def take_first_word(word):
        return word.split(" ")[0]

def model(df, car_predict):
    '''
    Predict the price of a car.

    Parameters:
    - Dataframe (pd.DataFrame): Merged Dataframe
    - Car (pd.DataFrame): Car we want to predict
    '''


    merged_df = df
    merged_df.rename(columns=lambda x: x.strip(), inplace=True)
    #
    # for column in merged_df.columns:
    #     print(column)


    # ask
    merged_df["car_model_small"] = merged_df["car_model"].map(take_first_word)
    merged_df.drop(columns="car_model", inplace=True)


    # Creating X
    X = merged_df.drop( columns=['car_code' , 'car_model_year',  'Next_YoY_Price', 'Next_YoY_Pr_Pred',
       'Price_sd_scaled' ])


    #Prepearing columns
    categorical_features = [] # intentionaly removing these features ['car_manufacturer', 'car_model_small']
    year_features = ["Year_x", "calendar_year" ]
    num_feat = ['Price_YoY','Price R','Doors','Settings_Pickup truck','Length between the axis','Sidewall height',
                'Weight','Car gearbox_Manual','Height','Maximum speed','Acceleration 0100 km/h in S',
                'Car payload','Width','Cylinder diameter','Fuel tank','Length','Year_y','Trunk','Road consumption',
                'Light in the trunk','Specific power','Piston course','Reader score','Provenance']

    # Impute then scale numerical values:
    num_inputer = Pipeline([
        ('imputer', SimpleImputer(strategy="mean"))
        #('standard_scaler', StandardScaler())
    ])
    standard_scaler = Pipeline([
        #('imputer', SimpleImputer(strategy="mean"))
        ('standard_scaler', StandardScaler())
    ])
    # Encode categorical values
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # Parallelize "num_transformer" and "cat_transfomer"
    preprocessor = ColumnTransformer([
        ('cat_transformer', cat_transformer, categorical_features), #,
        ('standard_scaler', standard_scaler, year_features) ,
        ('num_inputer', num_inputer, num_feat)  #numerical_columns
    ])

    # y (target)
    y = merged_df['Next_YoY_Price']


    X = preprocessor.fit_transform(X)  #keep in mind, is not procedural good

    # Split data into train, test and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3, random_state = 42  # TEST = 30%
    )

    # Use the same function above for the validation set
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size = 0.5, random_state = 42  # TEST = 15%
)

    model_file = "xgb_model.pkl"

    # Check if the model file exists
    if os.path.exists(model_file):
        # Load the model from the file
        with open(model_file, "rb") as f:
            xgb_regressor = pickle.load(f)
        print("Model loaded from", model_file)
    else:
        # Train the model
        xgb_regressor = xgb.XGBRegressor(max_depth=13, learning_rate=0.04, n_estimators=100)
        eval_set = [(X_train, y_train), (X_test, y_test)]
        xgb_regressor.fit(X_train, y_train, eval_set=eval_set, eval_metric="rmse", verbose=False)

        # Save the trained model to a file
        with open(model_file, "wb") as f:
            pickle.dump(xgb_regressor, f)
        print("Model trained and saved to", model_file)


    # Take second argument of the function(car that we want to predict)
    car_predict = car_predict.drop( columns=['car_code' , 'car_model_year',  'Next_YoY_Price', 'Next_YoY_Pr_Pred', 'Price_sd_scaled' ])
    # Preprocess
    car_transformed = preprocessor.transform(car_predict)
    # Predict
    prediction = xgb_regressor.predict(car_transformed)

    return prediction
