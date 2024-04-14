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
import re




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



def repeated_words(string_text):
    string_text = re.sub(r'\W', ' ', string_text)
    word_list = string_text.split()

    # use list comprehension to select items that appear more than 1 time
    repeated_words = [word for word in word_list if word_list.count(word) > 1]

    value_to_return = ' '.join(repeated_words).strip()
    return value_to_return

def unique_words(string_text):
    string_text = re.sub(r'\W', ' ', string_text)
    word_list = string_text.split()

    # use list comprehension to select items that appear only 1 time
    unique_words = [word for word in word_list if word_list.count(word) == 1]

    value_to_return = ' '.join(unique_words).strip()
    return value_to_return

@app.get("/car_name_predict/{car_name}")
def car_name_predict(car_name: str):
    # expecting car name as:  Peugeot-2008  I facelift (2016-2019) #==> Keep in mind: no slash / can be used!
    current_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    car_features_price_path = os.path.join(current_directory, 'raw_data', 'car_features_pr_pred_v2.csv')
    car_features_price_df = pd.read_csv(car_features_price_path)
    car_model_list_df = car_features_price_df[["car_manufacturer", "car_model", "car_model_year", "car_code"]].sort_values(by=['car_manufacturer', 'car_model', 'car_model_year'], ascending=[True, True, False])
    car_model_list_df.dropna(inplace=True)
    #we removed manufacturer from the name
    car_model_list_df['car_name'] =  car_model_list_df['car_model'] + ' ' + car_model_list_df['car_model_year'].astype(int).astype(str)
    car_model_list_df['car_name'] = car_model_list_df['car_name'].map(lambda x: x.lower())
    car_model_list_df['car_model_short'] = car_model_list_df['car_model'].map(lambda x: x.split()[0].lower() )

    car_name =  car_name.lower()


    #find years of the car mode
    # e.g Volvo-XC60 I facelift (2013-2017) => 2013-2017
    # find all 4 digit numbers
    numbers = re.findall(r'\b\d{4}\b', car_name)
    year_to_look_for = 0
    if len(numbers) >=1:
        year_to_look_for = int(numbers[-1]) #we always take the last year

    #we will add the car name to all the cars and check the one(s) that are the most similar
    car_model_list_df['car_name_temp'] = car_model_list_df['car_name'].map(lambda x: x.lower() + ' ' + car_name)
    car_model_list_df['repeated_words'] = car_model_list_df['car_name_temp'].map( lambda x: repeated_words(x.lower()) )
    car_model_list_df['repeated_words_len'] = car_model_list_df['repeated_words'].map(lambda x:  len(x))
    car_model_list_df['unique_words'] = car_model_list_df['car_name_temp'].map( lambda x: unique_words(x.lower()) )
    car_model_list_df['unique_words_len'] = car_model_list_df['unique_words'].map(lambda x:  len(x))
    car_model_list_df['diff_year'] = abs( car_model_list_df['car_model_year'] - year_to_look_for )




    #filtering the results


    if (len(car_name.split('-')) > 1 ):
        #filtering the same manufacturer. Note we need to lower the manufacturer.
        car_model_list_df['car_manufacturer'] = car_model_list_df['car_manufacturer'].map( lambda x: x.lower() )
        if len( car_model_list_df.loc[car_model_list_df['car_manufacturer'] == car_name.split('-')[0] ] ) > 0:
            car_model_list_df = car_model_list_df.loc[car_model_list_df['car_manufacturer'] == car_name.split('-')[0] ]
        #filtering the same model short name.
        if len( car_model_list_df.loc[car_model_list_df['car_model_short'] == car_name.lower().split('-')[1].split()[0] ] ) > 0:
            car_model_list_df =  car_model_list_df.loc[car_model_list_df['car_model_short'] == car_name.lower().split('-')[1].split()[0] ]




    car_model_list_df.sort_values(by=['repeated_words_len', 'unique_words_len', 'diff_year' ], ascending=[False, True, True], inplace=True)

    #if we don´t find any name from the car nor its manufacturer, we return no car found
    car_name_no_year = re.sub(r'\(.*?\)', '', car_name.lower())

    common_words = [item for item in re.split(r'\W+', car_model_list_df.head(1)["repeated_words"].values[0] ) if item in re.split(r'\W+', car_name_no_year ) ]

    print(common_words)

    if len(common_words) == 0 and car_model_list_df.head(1)["car_manufacturer"].values[0] not in re.split(r'\W+', car_name_no_year) :
        return "no car found"

    print('okoko')


    #return  car_model_list_df.head(10).to_dict('records')
    car_code = car_model_list_df[["car_code"]].head(1).values[0][0].tolist()
    print(car_code)
    return  car_predict(car_code)  #car_code
