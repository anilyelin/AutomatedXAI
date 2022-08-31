__author__ = "Anil Yelin"
import pandas as pd
from sklearn.model_selection import train_test_split

#hardcoded path to the datasets
PATH = "src/data/parkinsons.csv"

class Data():
    """this class handles the dataset
    by loading the csv file and preprocessing it 
    when necessary"""

    def __init__(self):
        try:
            self.dataset = pd.read_csv(PATH)
        except:
            print("Dataset not found or could not be loaded")

    def showHead(self):
        print(self.dataset.head())

    def splitDataset(self):
        data = self.data 
        target = data['status']
        features = data.drop(columns=['name','status'])
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, 
        random_state=42)
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    obj = Data()
    obj.showHead()