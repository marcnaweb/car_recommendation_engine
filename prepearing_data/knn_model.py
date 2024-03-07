import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os








# file_path = '/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/unique_car_models_df.csv'
# df = pd.read_csv(file_path)
# df.set_index('car_code')
# features_df = df.drop(columns=['car_model'])

# knn = NearestNeighbors(n_neighbors=6)

# knn.fit(features_df)



# # Use a car's features as our query point
# query = features_df.iloc[123].values.reshape(1, -1)

# # Find the 5 nearest neighbors to the first car (excluding itself)
# distances, indices = knn.kneighbors(query)

# # Get the car codes of the 6 closest matches (including the query point itself)
# closest_cars_indices = indices.flatten()
# closest_cars = df.iloc[closest_cars_indices][['car_code', 'car_model']]

# print(closest_cars)




def show_similar_cars(car_code):
    '''

    '''
    current_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # Define the relative paths to the CSV files
    features_relative_path = os.path.join(current_directory, 'raw_data', 'car_features_pr_pred_v2.csv')  #the same as the one used to consult price pred! :)
    features_df = pd.read_csv(features_relative_path)
    features_df.set_index('car_code', inplace=True)

    features_df = features_df.drop(columns=['car_model', 'car_manufacturer', 'car_model_year', 'price_pred'])



    knn = NearestNeighbors(n_neighbors=60) #change if needed
    knn.fit(features_df)
    # Use a car's features as our query point
    query = features_df.loc[car_code].values.reshape(1, -1)
    # Find nearest neighbors to the first car (excluding itself)
    distances, indices = knn.kneighbors(query)
    #print("distances are below")
    #print(distances)
    #print(features_df.iloc[indices[0] ].index.to_list() )
    closest_cars_indices = features_df.iloc[indices[0] ].index.to_list()
    return closest_cars_indices

    """
    current_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Define the relative paths to the CSV files
    features_relative_path = os.path.join(current_directory, 'raw_data', 'unique_car_models_df.csv')
    car_price_relative_path = os.path.join(current_directory, 'raw_data', 'car_prices_enriched_v2.csv')

    # Read the CSV files using the relative paths
    df = pd.read_csv(features_relative_path)

    #delte this if working
    # file_path = '/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/unique_car_models_df.csv'
    # df = pd.read_csv(file_path)

    features_df = df

    features_df.set_index('car_code', inplace=True)

    features_df = df.drop(columns=['car_model'])

    knn = NearestNeighbors(n_neighbors=12) #cahnge back to original = 6

    knn.fit(features_df)

    # Use a car's features as our query point
    query = features_df.loc[car_code].values.reshape(1, -1)

    # Find the 5 nearest neighbors to the first car (excluding itself)
    distances, indices = knn.kneighbors(query)

    # Get the car codes of the 6 closest matches (including the query point itself)
    closest_cars_indices = indices.flatten()
    #return this (closest_cars) if we need to return only car models
    closest_cars = df.iloc[closest_cars_indices][['car_model']]

    #NEW
    # now also returning manufacturer name with car model.
    car_code_list = closest_cars.index.tolist()
    # Take first raw dataframe where we have manufacturer names.
    with_manufacturer_df = pd.read_csv(car_price_relative_path)
    with_manufacturer_df.set_index('car_code', inplace=True)
    #return this if we need to return car model and car manufacturer names
    closest_cars_with_manufacturer = with_manufacturer_df.loc[car_code_list][['car_manufacturer','car_model']]

    return closest_cars_with_manufacturer
    """
