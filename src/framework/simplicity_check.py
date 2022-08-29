__author__ = "Anil Yelin"

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import shap

class SimplictyCheck():
    """this class implements the 
    Simplicity Checker of the automated 
    XAI explainability checker"""

    def __init__(self):
        pass