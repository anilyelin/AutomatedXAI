import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt 
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit.components.v1 as components

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
st.write("Shap values")
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


fig, ax = plt.subplots()
st.write("Shap values")
ax = shap.summary_plot(gbr_shap_values, X_train)
st.pyplot(fig)

st.subheader("Partial Dependence Plots")

features = ["fixed acidity","volatile acidity","citric acid","residual sugar",
       "chlorides","free sulfur dioxide","total sulfur dioxide","density","pH",
       "sulphates","alcohol"]
#option = st.selectbox("Select the feature for Partial Dependence Plot",features)

#fig, ax = plt.subplots()
#st.write("Dependence Plot")
#ax = shap.dependence_plot("alcohol", gbr_shap_values, X_train)
#st.pyplot(fig)


# get the prediction and put them with the test data
X_output = X_test.copy()
X_output.loc[:, 'predict'] = np.round(gbr.predict(X_output),2)

# randomly pick some observations
random_picks = np.arange(1,330,50) # every 50 rows
S = X_output.iloc[random_picks]

shap.initjs()

def shap_plot(j):
    explainerModel = shap.TreeExplainer(gbr)
    shap_values_Model = explainerModel.shap_values(S)
    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], S.iloc[[j]])
    return p

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


#st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))