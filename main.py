import numpy as np
import pandas as pd
from prepearing_data.data import clean_data
from prepearing_data.data import preprocess
from prepearing_data.pipeline import pipeline



df = pd.read_csv('/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/car_files_4c_en.csv') #change location

if __name__ == '__main__':
    data_cleaned = clean_data(df)
    data_preprocessed = preprocess(data_cleaned)
    pipeline_data = pipeline(data_preprocessed)
    print(pipeline_data)
    # print(data_preprocessed.dtypes) # just testing
