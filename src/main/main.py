import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt 
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import train_test_split


st.title("XAI Framework for evaluating explainability of black box ML-algorithms")

st.write("Below is datasets for the classification of wine regarding quality")

wine_df = pd.read_csv("/Users/anilyelin/Documents/Masterarbeit/AutomatedXAI/AutomatedXAI/src/data/winequality-red.csv")
st.write(wine_df.head())

st.subheader("Black Box Algorithm 1: RandomForestRegressor")

Y = wine_df['quality']
X = wine_df[["fixed acidity","volatile acidity","citric acid","residual sugar",
       "chlorides","free sulfur dioxide","total sulfur dioxide","density","pH",
       "sulphates","alcohol"]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
rfr = RandomForestRegressor(max_depth=6, n_estimators=10, random_state=0)
rfr.fit(X_train, y_train)

st.write("Random Forest Regressor Model Score:", rfr.score(X_test, y_test))

rfr_shap_values = shap.TreeExplainer(rfr).shap_values(X_train)

fig, ax = plt.subplots()

ax = shap.summary_plot(rfr_shap_values, X_train, plot_type='bar')
st.pyplot(fig)


fig, ax = plt.subplots()

ax = shap.summary_plot(rfr_shap_values, X_train)
st.pyplot(fig)


st.subheader("Black Box Algorithm 2: GradientBoostRegressor")

gbr = GradientBoostingRegressor(random_state=0, n_estimators=10)
gbr.fit(X_train, y_train)

st.write("Gradient Boost Regressor Score: ", gbr.score(X_test, y_test))


gbr_shap_values = shap.TreeExplainer(gbr).shap_values(X_train)
fig, ax = plt.subplots()

ax = shap.summary_plot(gbr_shap_values, X_train, plot_type='bar')
st.pyplot(fig)
