import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler




def pipeline(df: pd.DataFrame) -> pd.DataFrame:
    ''' Takes Dataframe, returns Dataframe (Binarized, onehotecnoded, scaled) after using pipeline'''
    #columns we want to encode with onehotencoder.
    columns_to_encode = ['Propulsion', 'Car gearbox', 'Fuel', 'Settings', 'Car size']
    #Saving numerical columns to impute them. (remove Nan's)
    numeric_columns = df.select_dtypes(include=['number'])
    numeric_column_names = numeric_columns.columns.tolist()

    #function to Binarize column. (transform from binary value to 0 and 1)
    def binarize_column(column):
        binarizer = LabelBinarizer()
        return binarizer.fit_transform(column)

    pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('onehot', OneHotEncoder(), columns_to_encode),  # Apply one-hot encoding to specified columns
        ('binarize', FunctionTransformer(binarize_column, validate=False), ['Provenance']),  # Binarize the binary column
        ('imputer', SimpleImputer(strategy='median'), numeric_column_names),  # Impute missing values
    ], remainder='passthrough')),
    ('scaler', MinMaxScaler()) #scale dataframe
])

    final_scaled_df = pipeline.fit_transform(df)
    final_scaled_df = pd.DataFrame(final_scaled_df)
    return final_scaled_df
