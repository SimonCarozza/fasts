import os
from os.path import dirname,  join
from pandas import read_csv


def load_data(data_file_name, dataframe=False):

    module_path = dirname(__file__)

    # print("module_path:", module_path)

    if data_file_name == "mediumposts.csv":
        date_column = "published"
        values = "posts"
    elif data_file_name == "passengers.csv":
        date_column = "Month"
        values = "Passengers"
    elif data_file_name == "sunspots.csv":
        date_column = "Month"
        values = "Sunspots"
    else:
        raise ValueError("valid values for 'data_file_name' are "
            "['mediumposts.csv', 'passengers.csv', 'sunspots.csv']")

    csv_file = join(module_path, "data", data_file_name)

    # print(csv_file)

    df = read_csv(
        csv_file, parse_dates=[date_column])
    if not dataframe: 
        # return series with DateTime series
        return df.set_index(date_column, drop=True)[values]
    else:
        return df
