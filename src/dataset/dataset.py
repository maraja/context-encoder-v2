import sys, os
import pandas

sys.path.append("../../")

default_data_path = os.path.join("..", "data")

class DatasetMixin:
    # save dataset and retrieve dataset information.
    # load datasets based on parameters.
    @staticmethod
    def get_datasets():
        df = pandas.read_csv(os.path.join(default_data_path, "datasets.csv"))
        print(df)

    @staticmethod
    def save_dataset():
        pass