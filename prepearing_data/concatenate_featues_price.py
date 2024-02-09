import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

from prepearing_data.data import clean_data
from prepearing_data.data import preprocess
from prepearing_data.pipeline import pipeline


# Cleaning Feature data
def get_cleaned_scaled_features_df(df:pd.DataFrame) -> pd.DataFrame:
    '''Take DataFrame as argument and return Cleaned,preprocesed,scaled df'''
    # clean
    df = clean_data(df)
    #preprocess
    df = preprocess(df)
    # transform with pipeline
    df = pipeline(df)
    return df


# Merging features and price df's in one merged df
def concatenate_features_prices_df(df:pd.DataFrame, df_1:pd.DataFrame) -> pd.DataFrame:
    '''
        Take 2 dataframes as arguments and merge in one.
        Return merged DataFrame
    '''
    merged_df = df_1.merge(df, left_on="car_code", right_on="car_code", how="left")
    #remove white space from some columns
    merged_df.rename(columns=lambda x: x.strip(), inplace=True)
    return merged_df

