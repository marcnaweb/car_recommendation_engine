import pandas as pd
from sklearn.neighbors import NearestNeighbors








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
    file_path = '/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/unique_car_models_df.csv'
    df = pd.read_csv(file_path)

    features_df = df

    features_df.set_index('car_code', inplace=True)

    features_df = df.drop(columns=['car_model'])

    knn = NearestNeighbors(n_neighbors=6)

    knn.fit(features_df)

    # Use a car's features as our query point
    query = features_df.loc[car_code].values.reshape(1, -1)

    # Find the 5 nearest neighbors to the first car (excluding itself)
    distances, indices = knn.kneighbors(query)

    # Get the car codes of the 6 closest matches (including the query point itself)
    closest_cars_indices = indices.flatten()
    closest_cars = df.iloc[closest_cars_indices][['car_model']]

    return closest_cars
