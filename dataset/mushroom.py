import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .ssl_dataset import BaseSSLDataset


class MushroomDataset(BaseSSLDataset):
    """
    All features are categorical data. We only need to convert them into
    a reasonable meaning which can calc the metric. For each unique value of
    one feature, we compute the proportion between edible and poisonous class
    only on training data.
    Class Distribution:
    edible:     4208 (51.8 %)
    poisonous:  3916 (48.2 %)
    total:      8124 instances
    """

    default_file_dir = "dataset/file/mushrooms.csv"
    data_header = [
        "class",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises?",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    ]

    def __init__(self, file_dir=default_file_dir):
        super().__init__()

        logging.info("MushroomData __init__")

        df = pd.read_csv(file_dir)
        # Map classes feature
        labelencoder = LabelEncoder()
        for column in df.columns:
            df[column] = labelencoder.fit_transform(df[column])

        self._mushroom_df = df

        self.y = self._mushroom_df["class"].to_numpy()
        self.X = self._mushroom_df.drop(["class"], axis=1).to_numpy()
        self.name = "Mushroom"


if __name__ == "__main__":
    dataset = MushroomDataset()

    print(dataset.get_data(40))
