import numpy as np
import pandas as pd

from prepearing_data.concatenate_featues_price import get_cleaned_scaled_features_df
from prepearing_data.concatenate_featues_price import concatenate_features_prices_df

#
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder



features_df = pd.read_csv('/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/car_files_4c_en.csv') #change location
car_prive_ready_df = pd.read_csv('/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/car_prices_w_prices_scaled.csv', index_col=0)

#not ready
df_car_price = pd.read_csv('/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/car_prices_enriched_v2.csv')





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

    numerical_columns = X.columns
    numerical_columns = numerical_columns.delete(0)
    numerical_columns = numerical_columns.delete(-1)
    # Impute then scale numerical values:
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy="mean"))
        #,('standard_scaler', StandardScaler())
    ])

    # Encode categorical values
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)




    # Parallelize "num_transformer" and "cat_transfomer"
    preprocessor = ColumnTransformer([
        ('cat_transformer', cat_transformer, ['car_manufacturer', 'car_model_small']), #, 'car_model'  --> was removed
        #('num_transformer', num_transformer, [ 'Year','Price_YoY'])
        ('num_transformer', num_transformer, numerical_columns )
    ])









    # data_cleaned = clean_data(features_df)
    # data_preprocessed = preprocess(data_cleaned)
    # pipeline_data = pipeline(data_preprocessed)


    #print(merged_df[merged_df.index == 87887])
    #print(nika)
