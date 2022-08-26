__author__ = "Anil Yelin"

from cmath import atan
from multiprocessing.spawn import prepare
import pandas as pd
import shap 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class RobustnessCheck():
    """this class implements the robustness
    check of the automated explainability 
    checker for black box algorithms"""
    def __init__(self):
        # loading the parkinson dataset
        self.data = pd.read_csv("src/data/parkinsons.csv")

    def prepareData(self):
        """preparing and preprocessing the dataset.
        This function uses the train_test_split method 
        from scikit learn to split the dataset into training
        and test sets.
        
        returns: X_train which is the training set
                 X_test which the test set
                 y_train which are the labels for training set
                 y_test which are the labels for the test set """
        #defining the target which is the column status in the datasets
        data = self.data
        target = data['status']
        #defining the features by dropping the columns name and status from the dataset
        features = data.drop(columns=['name','status'])
        X_train, X_test, y_train, y_test = train_test_split(features, 
        target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def trainModel(self):
        """this method will train the actual classifier. In this case
            we will use the ExtraTreesClassifier provided by scikit learn
            returns: model which the fitted ExtraTreesClassifier"""
        data = self.prepareData()
        X_train = data[0]
        X_test  = data[1]
        y_train = data[2]
        y_test = data[3]
        model = ExtraTreesClassifier(n_estimators=200, max_depth=30)
        model.fit(X_train, y_train)
        return model

    
    def createExplainer(self):
        """this method will initalize the SHAP Tree Explainer
           which is used for creating explanations to individual #
           data points """
        model = self.trainModel()
        data = self.prepareData()
        X_test = data[1]
        explainer = shap.TreeExplainer(model)
        #shap_values = explainer.shap_values(X_test)
        #shap_values = shap_values[0]
        return explainer 

    def getShapValues(self):
        """this method calculates the corresponding shap values
        given an shap explainer object

        returns: shap_values"""
        data = self.prepareData()
        X_test = data[1]
        explainer = self.createExplainer()
        shap_values = explainer.shap_values(X_test)
        shap_values = shap_values[0]
        return shap_values

    def explainIndividualInstance(self):
        """this method will use SHAP to 
        explain individual instances from the dataset by
        creating a shap force plot"""
        model = self.trainModel()
       

        data = self.prepareData()
        X_test = data[1]
        instance = X_test.loc[[138]]
        print(model.predict(X_test.loc[[138]]))

        explainer = self.createExplainer()
        shap_values = explainer.shap_values(instance)
        #instance = explainer.shap_values(instance)
        #a = shap.force_plot(explainer.expected_value[1], shap_values[1], instance, 
        #matplotlib=True)
        #plt.show(a)
        print("SHAP values for instance with index 138:")
        print()
        #print(shap_values)
        print()
        print("Expected values")
        print()
        print(explainer.expected_value[1])

        

    def test(self):
        shap_values = self.getShapValues()
        self.explainIndividualInstance()




if __name__ == "__main__":
    obj = RobustnessCheck()
    obj.prepareData()
    obj.test()