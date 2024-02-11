
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import os


def train_and_save_knn_model():
    file_path = '/Users/bididudy/code/marcnaweb/car_recommendation_engine/raw_data/unique_car_models_df.csv'
    df = pd.read_csv(file_path)

    model_dir = '/Users/bididudy/code/marcnaweb/car_recommendation_engine/models'  # Folder name where models will be saved

    # Create the directory if it does not exist
    os.makedirs(model_dir, exist_ok=True)

    features_df = df.set_index('car_code', inplace=False)
    features_df = features_df.drop(columns=['car_model'])

    knn = NearestNeighbors(n_neighbors=6)
    knn.fit(features_df)

    # Save the kNN model to the 'models' directory
    with open(os.path.join(model_dir, 'knn_model.pkl'), 'wb') as f:
        pickle.dump(knn, f)

    # Save the preprocessed features DataFrame to the 'models' directory
    with open(os.path.join(model_dir, 'features_df.pkl'), 'wb') as f:
        pickle.dump(features_df, f)

# Call the function to train and save your model
train_and_save_knn_model()
