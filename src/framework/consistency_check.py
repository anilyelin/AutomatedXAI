__author__ = "Anil Yelin"

import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split

class ConsistencyCheck():

    def __init__(self):
        self.data = pd.read_csv("src/data/parkinsons.csv")
        

    def show_data(self):
        print(self.data.head())


if __name__ == "__main__":
    obj = ConsistencyCheck()
    obj.show_data()