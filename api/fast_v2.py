from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import numpy as np
import pandas as pd

from prepearing_data.concatenate_featues_price import get_cleaned_scaled_features_df
from prepearing_data.concatenate_featues_price import concatenate_features_prices_df

from prepearing_data.model import model
from prepearing_data.knn_model import show_similar_cars
import os
import io




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

    #consult the car feature price pred csv file to check the car price prediction
    current_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    car_features_price_path = os.path.join(current_directory, 'raw_data', 'car_features_pr_pred_v2.csv')
    car_features_price_df = pd.read_csv(car_features_price_path)

    nearest_cars_codes  = show_similar_cars(car_code)  #note:: include the original car code on the top
    #print(nearest_cars_codes)

    #bring the given car_code to the first place
    nearest_cars_codes.pop(nearest_cars_codes.index(car_code)) #remove the original car code

    nearest_cars_features = car_features_price_df[car_features_price_df['car_code'].isin(nearest_cars_codes) ][['car_code',  'car_manufacturer', 'car_model','car_model_year','price_pred' ]]
    #drop duplicate manufacturer or model
    #nearest_cars_features.sort_values(by='car_manufacturer', inplace=True) #do not sort by manufacturer: cars are already sorted by distance from the knn model
    nearest_cars_features.drop_duplicates(subset=['car_manufacturer'], keep='first', inplace=True)

    original_car_features = car_features_price_df[car_features_price_df['car_code'] == car_code][['car_code',  'car_manufacturer', 'car_model','car_model_year','price_pred' ]]
    nearest_cars_features = nearest_cars_features[nearest_cars_features['car_manufacturer'] !=  original_car_features['car_manufacturer'].values[0] ]


    nearest_cars_features = pd.concat([original_car_features, nearest_cars_features])

    return    nearest_cars_features.head(10).to_dict('records')  #return maximum 10 cars


@app.get("/get_car_model_list/")
def get_car_model_list():
    current_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    car_features_price_path = os.path.join(current_directory, 'raw_data', 'car_features_pr_pred_v2.csv')
    car_features_price_df = pd.read_csv(car_features_price_path)
    car_model_list_df = car_features_price_df[["car_manufacturer", "car_model", "car_model_year", "car_code"]].sort_values(by=['car_manufacturer', 'car_model', 'car_model_year'], ascending=[True, True, False])
    car_model_list_df.dropna(inplace=True)
    car_model_list_df['car_model_year'] = car_model_list_df['car_model_year'].astype(int)

    stream = io.StringIO()
    car_model_list_df.to_csv(stream, index = False)

    response = StreamingResponse(iter([stream.getvalue()]),
                                media_type="text/csv"
                            )


    return response
    #in the front end, the response is retreived as a csv e.g.:
    #get_car_model_list_df = pd.read_csv("http://127.0.0.1:8000/get_car_model_list/")
    #get_car_model_list_df
