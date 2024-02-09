import numpy as np
import pandas as pd

from prepearing_data.concatenate_featues_price import get_cleaned_scaled_features_df
from prepearing_data.concatenate_featues_price import concatenate_features_prices_df


from prepearing_data.model import model

features_df = pd.read_csv('/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/car_files_4c_en.csv') #change location
car_prive_ready_df = pd.read_csv('/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/car_prices_w_prices_scaled.csv', index_col=0)


def price_diference(num):
    if num > 1:
        return print(f"price increased by {round(num - 1, 3)}%")
    elif num < 1:
        return print(f"price decreased by {round((num - 1)*-1, 3)}%")


if __name__ == '__main__':
    features_cleaned_df = get_cleaned_scaled_features_df(features_df)
    merged_df = concatenate_features_prices_df(features_cleaned_df ,car_prive_ready_df)

    # Specify a car which we want to predict
    car = merged_df[merged_df.index == 7780 ]

    # apply to our model
    model = model(merged_df, car)
    price_diference(model[0])
