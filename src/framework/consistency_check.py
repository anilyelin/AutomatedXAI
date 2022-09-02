__author__ = "Anil Yelin"

from re import S
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

    def explainIndividualInstance(self, k):
        """this method will use SHAP to 
        explain individual instances from the dataset by
        creating a shap force plot and calculating 
        the SHAP score
        param: 
            k = number of randomly drawn samples from the dataset
        return 
            SHAP scores for the k samples"""
    
        data = self.prepareData()
        X_test = data[1]
        X_test = X_test.sample(n=k)
        index = list(X_test.index)
        result = []
        for elem in index:
            instance = X_test.loc[[elem]]
            explainer = self.createExplainer()
            shap_values = explainer.shap_values(instance)
            shapScore = explainer.expected_value[1] + np.sum(shap_values[1])
            result.append(shapScore)
        #instance = explainer.shap_values(instance)
        #a = shap.force_plot(explainer.expected_value[1], shap_values[1], instance, 
        #matplotlib=True)
        #plt.show(a)
        # calculating the f(x) value which is basically the sum of the 
        # base value and the sum of the shap values for that particular instance
        print("SHAP Score for instance with [index 138] is: ", explainer.expected_value[1]+np.sum(shap_values[1]))
        return result 

    @staticmethod
    def main():
        obj = ConsistencyCheck()
        obj.createExplainer()
        #print(obj.getShapValues())
        print(obj.explainIndividualInstance(3))


if __name__ == "__main__":
    ConsistencyCheck().main()