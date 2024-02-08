import numpy as np
import pandas as pd

from prepearing_data.concatenate_featues_price import get_cleaned_scaled_features_df
from prepearing_data.concatenate_featues_price import concatenate_features_prices_df

#
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




features_df = pd.read_csv('/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/car_files_4c_en.csv') #change location
car_prive_ready_df = pd.read_csv('/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/car_prices_w_prices_scaled.csv', index_col=0)

#not ready
df_car_price = pd.read_csv('/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/car_prices_enriched_v2.csv')


def train_test_split_fun():
    pass


if __name__ == '__main__':
    features_cleaned_df = get_cleaned_scaled_features_df(features_df)
    merged_df = concatenate_features_prices_df(features_cleaned_df ,car_prive_ready_df)
    merged_df.rename(columns=lambda x: x.strip(), inplace=True)


    X = merged_df.drop( columns=['car_code' , 'car_model_year',  'Next_YoY_Price', 'Next_YoY_Pr_Pred',
       'Price_sd_scaled' ])

    def take_first_word(word):
        return word.split(" ")[0]

    X["car_model_small"] = X["car_model"].map(take_first_word)
    X.drop(columns="car_model", inplace=True)
    y = merged_df['Next_YoY_Price']

    num_feat = [feature for feature in X.select_dtypes(include='number').columns.tolist() if feature not in ["Year_x", "calendar_year" ] ]
    #num_feat
    categorical_features = list(X.select_dtypes(include='object').columns)
    categorical_features = [] # intentionaly removing these features ['car_manufacturer', 'car_model_small']
    year_features = ["Year_x", "calendar_year" ]




    # Impute then scale numerical values:
    num_inputer = Pipeline([
        ('imputer', SimpleImputer(strategy="mean"))
        #('standard_scaler', StandardScaler())
    ])

    standard_scaler = Pipeline([
        #('imputer', SimpleImputer(strategy="mean"))
        ('standard_scaler', StandardScaler())
    ])

    # Encode categorical values
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Parallelize "num_transformer" and "cat_transfomer"
    preprocessor = ColumnTransformer([
        ('cat_transformer', cat_transformer, categorical_features ), #, 'car_model'  --> was removed
        ('standard_scaler', standard_scaler, year_features ) ,
        ('num_inputer', num_inputer, num_feat  )  #numerical_columns
    ])


    X = preprocessor.fit_transform(X)  #keep in mind, is not procedural good

    # Split data into train, test and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3, random_state = 42  # TEST = 30%
    )

    # Use the same function above for the validation set
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size = 0.5, random_state = 42  # TEST = 15%
    )
    print(X.shape)
    print(X_train.shape)
    print(X_val.shape)





    # data_cleaned = clean_data(features_df)
    # data_preprocessed = preprocess(data_cleaned)
    # pipeline_data = pipeline(data_preprocessed)


    #print(merged_df[merged_df.index == 87887])
    #print(nika)
