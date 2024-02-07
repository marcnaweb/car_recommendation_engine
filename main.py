import numpy as np
import pandas as pd
from prepearing_data.data import clean_data
from prepearing_data.data import preprocess



df = pd.read_csv('/home/nika/code/marcnaweb/car_recommendation_engine/raw_data/car_files_4c_en.csv') #change location

if __name__ == '__main__':
    data_cleaned = clean_data(df)
    data_cleaned = preprocess(data_cleaned)
    # print(data_cleaned.dtypes['Reader score']) # just testing
    test = pd.DataFrame({'Column Name': data_cleaned.columns, 'Data Type': data_cleaned.dtypes})
    print(test.head(36))
    print(test.tail(36))
