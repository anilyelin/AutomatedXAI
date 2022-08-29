__author = "Anil Yelin"

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import shap 


class StabilityCheck():
    """this class implements 
    the stability check of the 
    automated XAI framework """
    def __init__(self):
        pass