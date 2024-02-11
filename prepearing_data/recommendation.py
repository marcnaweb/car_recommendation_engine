import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import os

def show_similar_cars(car_code):

    # Specify the directory where your models are saved
    model_dir = '/Users/bididudy/code/marcnaweb/car_recommendation_engine/models'  # Folder name where models are saved

    # Initialize features_df to None or some default
    features_df = None

    # Attempt to load features_df, ensuring it happens before any use
    features_df_path = os.path.join(model_dir, 'features_df.pkl')
    with open(features_df_path, 'rb') as f:
        features_df = pickle.load(f)

    # Now, ensure features_df is actually loaded before proceeding
    if features_df is None:
        raise Exception("Failed to load features DataFrame.")

    # Construct the path to each pickle file
    knn_model_path = os.path.join(model_dir, 'knn_model.pkl')
    features_df_path = os.path.join(model_dir, 'features_df.pkl')

    # Load the kNN model from the 'models' directory
    with open(knn_model_path, 'rb') as f:
        knn = pickle.load(f)

    # Load the preprocessed features DataFrame from the 'models' directory
    with open(features_df_path, 'rb') as f:
        features_df = pickle.load(f)

    # Use the car's features as our query point
    query_df = features_df.loc[[car_code]]

    # Find the 5 nearest neighbors to the car
    distances, indices = knn.kneighbors(query_df)

    # Get the car codes of the 6 closest matches (including the query point itself)
    closest_cars_indices = indices.flatten()
    closest_cars = features_df.iloc[closest_cars_indices].reset_index()[['car_code']]

    # Assuming you need to join back with the original df to get car_model names
    df = pd.read_csv('/Users/bididudy/code/marcnaweb/car_recommendation_engine/raw_data/unique_car_models_df.csv')
    closest_cars_with_models = pd.merge(closest_cars, df[['car_code', 'car_model']], on='car_code', how='left')
    #closest_cars_with_models.set_index('car_code', inplace=True)

    return closest_cars_with_models
