import numpy as np
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Receive dataframe. remove Nan's, remove columns that are usless """
    #Too many features with NaN's, remove columns which have more then 15000 Nan'n in column
    filtered_columns = df.columns[df.isna().sum() < 15000]
    filtered_df = df[filtered_columns]

    #remove some rows with a lot of Nan's
    test_filtered_df = filtered_df.dropna(subset='Acceleration')

    #columns that are usless.
    column_to_remove = ['Unnamed: 0.1','Assistance', 'Aspiration', 'Rear tires', 'Spare tire', 'Front tires',
                    'Urban autonomy', 'Generation', 'Front suspension', 'Rear suspension', 'Coupling', 'Valve command',
                    'Places', 'Disposition', 'Cylinders', 'Elastic element', 'Ipva R', 'Frontal area A', 'Engine code',
                    'Traction', 'Installation', 'Road autonomy', 'Engine power supply', 'Engine control activation',
                    'Gear change code', 'Corrected frontal area', 'Platform']

    # remove usless columns
    new_filtered_df = test_filtered_df.drop(columns=column_to_remove)
    #return clean dataframe
    return new_filtered_df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    ''' Receive dataframe. Transform values to numbers '''

    # columns that will be transformed same: Nan = 0 / standart equipment, optinal equipment = 1
    columns_to_transform = ['Hot air', 'Rev counter', 'Assisted direction', 'ABS brakes', 'Rear window', 'Central locking of the doors',
                        'Headrest for all occupants', 'Electric rearview mirror adjustment', 'Air conditioning',
                        'Bluetooth connection', 'Frontal Airbags', 'Steering wheel adjustment height',
                        'Electric front window control', 'Multifunctional steering wheel', 'Driver s seat with height adjustment',
                        'On board computer', 'Light in the trunk', 'Alloy wheels', 'USB connection', 'Radio',
                        'Folding rear seat', 'Perimeter anti theft alarm', 'Cooling liquid thermometer']
    df[columns_to_transform] = df[columns_to_transform].applymap(lambda x: 0 if pd.isnull(x) else 1)

    ########################################Preprocess [Acceleration]#################################################################
    # Preprocess Acceleration (0100 km/h 3,8 s = 3.8)
    df['Acceleration'] = df['Acceleration'].str.extract(r'(\d+\,\d+)')
    # Replace ',' with '.' and convert to numeric
    df['Acceleration'] = df['Acceleration'].str.replace(',', '.').astype(float)
    # Rename Column
    df = df.rename(columns={'Acceleration': 'Acceleration 0100 km/h in S'})

    ##########################################Preproces values with mm/kg/cm/ ETC ####################################################
    #function to remove strings (mm/kg/cm/) and transform to floats
    def extract_float_value(value):
        try:
            if isinstance(value, float):
                return value
            else:
                float_value = value.split()[0].replace(',', '.')
                return float(float_value)
        except (ValueError, IndexError):
            return np.nan

    #transforming
    for column in df[['Weight/Torque', 'Weight', 'Weight/power', 'Max power regime.', 'Cylinder diameter',
                      'Fuel tank', 'Specific power', 'Maximum power', 'Length', 'Maximum torque', 'Width', 'Height',
                      'Specific torque', 'Minimum height from the ground', 'Piston course', 'Front gauge', 'Displacement',
                      'Turns diameter', 'Rear gauge', 'Length between the axis', 'Maximum speed', 'Road consumption',
                      'Max torque regime', 'Car payload', 'Sidewall height', 'Unit displacement', 'Trunk', 'Urban']]:
        df[column] = df[column].apply(extract_float_value)
    #################################################################################################################################


    return df
