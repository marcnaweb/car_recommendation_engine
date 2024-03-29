from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd

from prepearing_data.concatenate_featues_price import get_cleaned_scaled_features_df
from prepearing_data.concatenate_featues_price import concatenate_features_prices_df

from prepearing_data.model import model
from prepearing_data.knn_model import show_similar_cars
import os




app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

features_pickle_file_path = "features_cleaned_df.pickle"

def load_or_create_features_df():
    if os.path.exists(f"features_cleaned_df.pickle"):
        # If the pickle file exists, load the DataFrame from it
        features_cleaned_df = pd.read_pickle(features_pickle_file_path)
        print("Data loaded !")
    else:
        # If the pickle file doesn't exist, create the cleaned and scaled features DataFrame
        current_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        features_relative_path = os.path.join(current_directory, 'raw_data', 'car_files_4c_en.csv')
        features_df = pd.read_csv(features_relative_path)
        features_cleaned_df = get_cleaned_scaled_features_df(features_df)
        # Save the DataFrame to a pickle file
        features_cleaned_df.to_pickle(features_pickle_file_path)
    return features_cleaned_df


@app.get("/")
def root():
    return { 'greeting': 'Hello' }

@app.get("/car_predict/{car_code}")
def car_predict(car_code: int):

    # Get the current directory of the script
    current_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Define the relative paths to the CSV files
    #features_relative_path = os.path.join(current_directory, 'raw_data', 'car_files_4c_en.csv')
    prices_relative_path = os.path.join(current_directory, 'raw_data', 'car_prices_w_prices_scaled.csv')

    # Read the CSV files using the relative paths
    #features_df = pd.read_csv(features_relative_path)
    car_prive_ready_df = pd.read_csv(prices_relative_path, index_col=0)


    #features_cleaned_df = get_cleaned_scaled_features_df(features_df)
    features_cleaned_df = load_or_create_features_df()
    merged_df = concatenate_features_prices_df(features_cleaned_df ,car_prive_ready_df)
    #Car we want to predict
    car = merged_df[merged_df['car_code'] == car_code ]

    #answer = predict cars devaluation
    answer = model(merged_df, car)
    final_car_predict_answer = answer[0]

    #Predict 5 nearest cars.
    #if we return this, we will have also car_codes.
    five_nearest_cars  = show_similar_cars(car_code)

    #create dictionary with with car_manufacturer as a key, and car_model as a value
    # manufacturer_model_dict = {}

    # for car_code, row in five_nearest_cars.iterrows():
    #     manufacturer_model_dict[row['car_manufacturer']] = row['car_model']
    #     #########################################################################

    # # original car that we are predicting
    # original_car = five_nearest_cars.iloc[0][['car_manufacturer','car_model']]
    # original_car_dict = {}
    # original_car_dict[original_car['car_manufacturer']] = original_car['car_model']

    # ###############################################################################



    # return {"Original_car": original_car_dict,
    #         "prediction": float(final_car_predict_answer),
    #         "similar_cars": manufacturer_model_dict,
    #         "similar_car_codes": five_nearest_cars['car_model']
    #     }

    #load new data with prediction
    prediction = os.path.join(current_directory, 'raw_data', 'car_features_pr_pred.csv')
    prediction_df = pd.read_csv(prediction)

    manufacturer_model_dict = {}
    added_manufacturers = set()

    for car_code, row in five_nearest_cars.iterrows():
        manufacturer = row['car_manufacturer']
        if manufacturer not in added_manufacturers:
            manufacturer_model_dict[car_code] = {'car_manufacturer': manufacturer, 'car_model': row['car_model']}
            added_manufacturers.add(manufacturer)

    # original car that we are predicting
    original_car = five_nearest_cars.iloc[0]
    original_car_dict = {'car_manufacturer': original_car['car_manufacturer'], 'car_model': original_car['car_model']}

    similar_cars_codes = []
    for car_code, data in manufacturer_model_dict.items():
        price_pred = prediction_df.loc[car_code, 'price_pred']
        car_year = prediction_df.loc[car_code, 'car_model_year']
        similar_cars_codes.append({"car_code": car_code, "car_manufacturer": data['car_manufacturer'], "car_model": data['car_model'], "price_pred": price_pred,"car_year":car_year })

    return {
        "Original_car": original_car_dict,
        "prediction": float(final_car_predict_answer),
        "similar_cars_codes": similar_cars_codes
    }
