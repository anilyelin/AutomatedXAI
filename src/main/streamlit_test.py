__author__ = "Anil Yelin"
__version__= "0.3"

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt 
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
from numpy import linalg as LA
from scipy import spatial

#utility function
@st.cache
def convert_df(df):
    """this function converts a pandas dataframe into a 
       csv file with UTF-8 encoding.
       params: 
             pandas dataframe
       returns:
             a csv file with UTF-8 encoding
    """
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def automatedChange(df, index):
    """this function will apply automated
    marginal changes to a particular row in a pandas 
    dataframe 
        input: 
              df: pandas dataframe
              index: integer to specify the row
        returns:
              modified pandas dataframe with respect to that row"""
    row_mean = df.loc[index].mean()
    row_std = df.loc[index].std()
    noise = np.random.normal(0.0, 0.5, len(df.columns))
    tmp = noise+df.loc[index]
    df.loc[index, df.columns] = [df.loc[index][i]+noise[i] for i in range(len(df.columns))]
    return df

st.title("Automated Explainability Checker Framework v0.3")
st.text("This streamlit app is a prototype for the proposed explainability framework\n"
"proposed in my master thesis")
st.header("Dataset Overview")
st.caption("Parkinson Dataset")
df = pd.read_csv("/Users/anilyelin/Documents/Masterarbeit/AutomatedXAI/AutomatedXAI/src/data/parkinsons.csv")

st.write(df.head())
csvFile = convert_df(df)
st.download_button(label="Download as csv file",data=csvFile, file_name="parkinsons.csv")
#############################################################################################

st.header("Model Training")
randomForest_tab, extraTrees_tab = st.tabs(["Random Forest Classifier", "Extra Trees Classifier"])

with randomForest_tab:
    blackBoxModels = ["Random Forest Classifier","Extra Trees Classifier"]
    modelChoice = st.selectbox("Please choose the first black box model for training",blackBoxModels)
    st.subheader("Random Forest Classifier")
    n_estimators = st.number_input("Please enter the number of estimators", min_value=10, step=1, value=300)
    random_state = st.number_input("Please enter a random state number", min_value=0, step=1, value=42)
    test_size = st.number_input("Please enter the size of the test dataset", min_value=0.1, max_value=0.4, value=0.2)
    max_depth = st.number_input("Please enter the max depth for a tree", min_value=10, value=30)
    st.caption("Hyperparameter Summary")
    data = df
    target = data['status']
    features = data.drop(columns=['status','name'])
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)

    rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    rfc.fit(X_train, y_train)
    #st.write("Model score for black box model: ",modelChoice," is ", rfc.score(X_test, y_test))
    rf_col1, rf_col2, rf_col3, rf_col4, rf_col5, rf_col6 = st.columns(6)
    rf_col1.metric("#Estimators", n_estimators)
    rf_col2.metric("Random State", random_state)
    rf_col3.metric("Tree Max Depth", max_depth)
    rf_col4.metric("Test Size", round(test_size,4))
    rf_col5.metric("Training Size", round(1-test_size,4))
    rf_col6.metric("Model Score", round(rfc.score(X_test, y_test),4))
    

with extraTrees_tab:
    st.subheader("Extra Trees Classifier")
    modelChoice1 = st.selectbox("Please choose a black box model for training",blackBoxModels,index=1,disabled=True)
    #st.write("\nPlease note that the entered hyperparameters will be same for both models.\n"
    #"The values cannot be changed for the second model!")
    st.warning('To ensure comparability between the models the hyperparameters are same for both models. They cannot be changed.', icon="⚠️")
    n_estimators1 = st.number_input("the number of estimators", min_value=n_estimators, step=1,disabled=True)
    random_state1 = st.number_input(" random state number", min_value=random_state, step=1,disabled=True)
    test_size1 = st.number_input("the size of the test dataset", min_value=test_size, max_value=0.4, disabled=True)
    max_depth1 = st.number_input("the max depth for a tree", min_value=max_depth, disabled=True)
    st.caption("Hyperparameter Summary")
    data = df
    target = data['status']
    features = data.drop(columns=['status','name'])
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state)
    etc = ExtraTreesClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    etc.fit(X_train, y_train)
    #st.write("Model score for black box model: ", modelChoice1, " is ", etc.score(X_test, y_test))
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("#Estimators", n_estimators1)
    col2.metric("Random State", random_state1)
    col3.metric("Tree Max Depth", max_depth1)
    col4.metric("Test Size", round(test_size,4))
    col5.metric("Training Size", round(1-test_size,4))
    col6.metric("Model Score", round(etc.score(X_test, y_test),4))

########################################################################################

st.header("Explainability Section")
st.write("This prototype will make use of SHAP to calculate the Shapley values for the features in the dataset\n"
"Shapley values will measure the marginal contribution to the outcome of the model")
st.latex(r'''
        \phi_i(p) = \sum_{S\subseteq N \setminus i}\frac{|S|!(|N|-|S|-1)!}{|N|} (v(S \cup \{i\}-v(S)))
     ''')

st.write("""SHAP is based on a strong mathematical foundation with several properties such as additivity,
null player and symmetry. In the thesis can be found a more detailed explanation on the theoretical
foundations of SHAP""")

st.subheader("Feature Importance based on SHAP values")

st.write("""The following bar chart below is showing the most important features according .
It should not be confused with Permutation Feature Importance which is a part of the 
proposed framework.""")

rfc_shap_values = shap.TreeExplainer(rfc).shap_values(X_train)
fig, ax = plt.subplots()

ax = shap.summary_plot(rfc_shap_values, X_train, plot_type='bar')
st.pyplot(fig)


#shap_values = explainer.shap_values(instance)

st.title("Explainability Checker Framework")
st.text("""The following figure is showing the architecture of the explainability checker framework.
There are in total five components. Both black box models will be analysed with respect
to the components. The implementation of the framework component is the upcoming section.
In each component there will be one black box model which will perform better in terms
of explainability.""")
st.image("method.png", width=250, caption="Architecture Overview")



###############################################################################################################
indexValue = list(X_test.index) #all index values of the X_test set will be stored in this list
#shap values for random forest classifier
explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(X_test)
shap_values = shap_values[0]

#shap values for extra trees classifier
explainer1 = shap.TreeExplainer(etc)
shap_values1 = explainer1.shap_values(X_test)
shap_values1 = shap_values1[0]


st.header("Explainability Checker Framework")
st.write("""In the following tabs every can component of the proposed explainability 
framework can be used for checking and evaluating the explainability of black box
models with corresponding data""")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Component Consistency", "Component Robustness", 
"Component Stability","Component Simplicity","Component Feature Importance"])

with tab1:
    st.subheader("Framework Component - Consistency")
    expander1 = st.expander("See explanation")
    expander1.write("""The consistency component will check local explanations of the two black box
    models and compare them against each other. The consistency check will be done for
    k datapoints. The key assumption is that the explanations for a given data instance
    should deviate with respect to given theta threshold""")

    kNumber = st.number_input("Please enter a value for parameter k", min_value=1, max_value=len(X_test)-1, step=1)
    consistencyThreshold = st.number_input("Please enter a value for threshold theta", min_value=0.01, max_value=1.0, step=0.01)
    distanceMeasure = st.selectbox("Choose distance measure",['Euclidean Distance','Cosine Similarity'])
    st.write("Distance Measure is: ", distanceMeasure)
    st.write("Entered number k is ",kNumber)
    st.write("The following table is showing the k data instances with its corresponding values")
    st.write(X_test.head(kNumber))
    # storing the results in a list for the table
    tableEuclidean = []
    tableTheta = []
    with st.spinner('Wait for it...'):
        for i in range(kNumber):
            instance = X_test.loc[[indexValue[i]]]
            shap_values = explainer.shap_values(instance)
            #extra trees
            shap_values1 = explainer1.shap_values(instance)
            #
            st.write("**"+modelChoice+"** SHAP Force Plot for instance with index: ", indexValue[i])
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], instance))
            #
            st.write("**Extra Trees Classifier** SHAP force plot for instance with index ", indexValue[i])
            st_shap(shap.force_plot(explainer1.expected_value[1], shap_values1[1], instance))
            fx1 = explainer.expected_value[1]+np.sum(shap_values[1])
            fx2 = explainer1.expected_value[1]+np.sum(shap_values1[1])
            consistencyDifference = round(np.abs(fx1-fx2),4)
            #calculate euclidean distance between shap vectors
            consistencyEuclideanDistance = np.round(LA.norm(shap_values[1]-shap_values1[1]),4)
            #calculate cosine similarity
            cosineSimilarity = 1-np.dot(shap_values[1],shap_values1[1].T)/(LA.norm(shap_values[1])*(LA.norm(shap_values1[1])))
        
            if distanceMeasure == "Euclidean Distance":
                tab1_col1, tab1_col2 = st.columns(2)
                #tab1_col1.metric(label="Explanation Difference", value=consistencyDifference)
                #storing results in a table
                tableEuclidean.append(consistencyEuclideanDistance)
                tableTheta.append(round(consistencyThreshold-consistencyEuclideanDistance,4))
                tab1_col1.metric(label="Euclidean Distance", value=consistencyEuclideanDistance)
                tab1_col2.metric(label="Threshold Delta", value=round(consistencyThreshold-consistencyEuclideanDistance,4))
            else:
                tab1_col1, tab1_col2 = st.columns(2)
                #tab1_col1.metric(label="Explanation Difference", value=consistencyDifference)
                tab1_col1.metric(label="Cosine Similarity", value=np.round(cosineSimilarity,4))
                tab1_col2.metric(label="Threshold Delta", value=round(consistencyThreshold-consistencyEuclideanDistance,4))

            tab1_difference = consistencyThreshold-consistencyEuclideanDistance
            if tab1_difference >= 0:
                st.success("Threshold is maintained")
            else:
                st.error("Threshold is not maintained")

            st.write("*******************************************************************************************")
        st.success('Calculation of SHAP values for the k instances was successful!')
    #for i in range(kNumber):
    #    instance = X_test.loc[[indexValue[i]]]
    #    shap_values = explainer.shap_values(instance)
    #    st.write("++"+modelChoice,"++SHAP Force Plot for instance with index: ", indexValue[i])
    #    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], instance))

    if st.button(label="Create Summary Table for Results"):
        st.subheader("Summary Table")
        st.write("Below you can see a table with all results with respect to your parameters.")
        euclideanDF = pd.DataFrame(tableEuclidean)
        thetaDF = pd.DataFrame(tableTheta)
        euclideanDF.columns = ["Euclidean Distance SHAP Vectors: RFC <--> ETC"]
        thetaDF.columns = ["Threshold Delta "+str(consistencyThreshold)]
        df_col_merged = pd.concat([euclideanDF, thetaDF], axis=1)
        st.write(df_col_merged)
        tableFile = convert_df(df_col_merged)
        st.download_button(label="Download results as csv file",data=tableFile, file_name="result_table.csv")
        st.write("Threshold of: ", consistencyThreshold, " is not maintained for ",int(thetaDF.lt(0).sum()) , " instances of ", kNumber, " instances (in total)")


with tab2:
    st.subheader("Framework Component - Robustness")
    expander2 = st.expander("See explanation")
    expander2.write("""The component robustness will analyse the explanations for specific data instances
    when marginal changes for some of the features are applied. The key assumption for this component 
    is that for marginal changes in the input data the corresponding explanation after the change shoould
    also be marginal.""")
    robustnessKNumber = st.number_input("Please enter a value for k",min_value=1, max_value=len(X_test)-1, step=1)
    st.write("Entered number k is ",robustnessKNumber)
    robustnessThreshold = st.number_input("Please enter a threshold value", min_value=0.1, max_value=0.5, step=0.01)
    st.write("The following table is showing the k data instances with its corresponding values")
    st.write(X_test.head(robustnessKNumber))
    X_test_copy = X_test.copy(deep=True)
    st.info("You can either make manual changes or use the provided button for automated marginal changes")
    if st.button("Apply automated marginal changes"):
        st.write("Calling function for automated marginal changes")
        automatedChange(X_test, 138)
        st.success("Marginal changes applied successfully!")
    
    st.subheader("Marginal Changes")
    cols = list(X_test.columns)
    st.write(cols[0])
    c0 = st.number_input("Marginal Change for feature: "+cols[0])
    st.write(cols[1])
    c1 = st.number_input("Marginal Change for feature: "+cols[1])
    st.write(cols[2])
    c2 = st.number_input("Marginal Change for feature: "+cols[2])
    st.write(cols[3])
    c3 = st.number_input("Marginal Change for feature: "+cols[3])
    st.write(cols[4])
    c4 = st.number_input("Marginal Change for feature: "+cols[4])
    st.write(cols[5])
    c5 = st.number_input("Marginal Change for feature: "+cols[5])
    st.write(cols[6])
    c6 = st.number_input("Marginal Change for feature: "+cols[6])
    st.write(cols[7])
    c7 = st.number_input("Marginal Change for feature: "+cols[7])
    st.write(cols[8])
    c8 = st.number_input("Marginal Change for feature: "+cols[8])
    st.write(cols[0])
    c9 = st.number_input("Marginal Change for feature: "+cols[9])
    st.write(cols[10])
    c10 = st.number_input("Marginal Change for feature: "+cols[10])
    st.write(cols[11])
    c11 = st.number_input("Marginal Change for feature: "+cols[11])
    st.write(cols[12])
    c12 = st.number_input("Marginal Change for feature: "+cols[12])
    st.write(cols[13])
    c13 = st.number_input("Marginal Change for feature: "+cols[13])
    st.write(cols[14])
    c14 = st.number_input("Marginal Change for feature: "+cols[14])
    st.write(cols[15])
    c15 = st.number_input("Marginal Change for feature: "+cols[15])
    st.write(cols[16])
    c16 = st.number_input("Marginal Change for feature: "+cols[16])
    st.write(cols[17])
    c17 = st.number_input("Marginal Change for feature: "+cols[17])
    st.write(cols[18])
    c18 = st.number_input("Marginal Change for feature: "+cols[18])
    st.write(cols[19])
    c19 = st.number_input("Marginal Change for feature: "+cols[19])
    st.write(cols[20])
    c20 = st.number_input("Marginal Change for feature: "+cols[20])
    st.write(cols[21])
    c21 = st.number_input("Marginal Change for feature: "+cols[21])

    st.write("Resulting changes of data instance with index: ")
    
    deltas = [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21]
    for i in range(22):
        X_test.loc[X_test.index[0], [cols[i]]] = [(X_test.iloc[0][cols[i]])+deltas[i]]
    #X_test.loc[X_test.index[0], [cols[1]]] = [(X_test.iloc[0][cols[1]])+c1]
    #X_test.loc[X_test.index[0], [cols[2]]] = [(X_test.iloc[2][cols[2]])+c2]
    #X_test.loc[X_test.index[0], [cols[3]]] = [(X_test.iloc[3][cols[3]])+c3]
    st.write(X_test.head(robustnessKNumber))
    #copy
    instance_robustness_copy = X_test_copy.loc[[X_test.index[0]]]
    shap_values_robustness_copy = explainer.shap_values(instance_robustness_copy)
    #marginal change
    instance_robustness = X_test.loc[[X_test.index[0]]]
    shap_values_robustness = explainer.shap_values(instance_robustness)
    st.write("Orignal Explanation")
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values_robustness_copy[1], instance_robustness_copy))
    st.write("Explanation after marginal changes")
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values_robustness[1], instance_robustness))
    shapScoreOrig = explainer.expected_value[1]+np.sum(shap_values[1])
    shapScoreMod = explainer.expected_value[1]+np.sum(shap_values_robustness[1])
    shapScoreDiff = np.round(np.abs(shapScoreOrig-shapScoreMod),4)
    tab2_col1, tab2_col2 = st.columns(2)
    tab2_col1.metric("Euclidean Distance",shapScoreDiff)
    tab2_col2.metric("Delta Value", np.round(shapScoreDiff-robustnessThreshold,4))

with tab3:
    st.subheader("Framework Component - Stability")
    expanderComponent3 = st.expander("See explanation")
    expanderComponent3.write("""The component stability will analyse the explanations of 
    neighboring data points. The key assumption is that for neighboring data points the 
    explanations should also similar""")
    tab3_kValue = st.number_input("Please enter the value k for neighboring data instances", min_value=1, max_value=3, value=3, step=1, disabled=True)
    tab3_theta = st.number_input("Please enter the threshold value theta", min_value=0.01, max_value=0.5, step=0.01)
    st.subheader("Stability Check for 111 & 112")
    st.write(X_test.loc[[111,112]])
    tmpDF = X_test.loc[[111,112]].copy()
    deltaDF = tmpDF.diff()
    deltaDF = deltaDF.tail(1)
    deltaDF.index.rename("Delta", inplace=True)
    st.write("Delta between feature values of row 111 and row 112")
    st.write(deltaDF)

    instance_111 = X_test.loc[[111]]
    shap_values_111 = explainer.shap_values(instance_111)
    #extra trees
    #shap_values1 = explainer1.shap_values(instance)
        
    st.write("RFC - SHAP Force Plot for instance 111")
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values_111[1], instance_111))
    st.write("RFC - SHAP Force Plot for instance 112")
    instance_112 = X_test.loc[[112]]
    shap_values_112 = explainer.shap_values(instance_112)
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values_112[1], instance_112))
    shap_values_111_etc = explainer1.shap_values(instance_111)
    shap_values_112_etc = explainer1.shap_values(instance_112)
    st.write("ETC - SHAP Force Plot for instance 111")
    st_shap(shap.force_plot(explainer1.expected_value[1], shap_values_111_etc[1], instance_111))
    st.write("ETC - SHAP Force Plot for instance 112")    
    st_shap(shap.force_plot(explainer1.expected_value[1], shap_values_112_etc[1], instance_112))
    tab3_col1, tab3_col2 = st.columns(2)
    stabilityDistance = np.round(LA.norm(shap_values_111[1]-shap_values_112[1]),4)
    #tab3 delta
    tab3_delta = tab3_theta-stabilityDistance
    tab3_col1.metric(label="Euclidean Distance", value=stabilityDistance)
    tab3_col2.metric(label="Threshold Value", value=round(tab3_delta,4))
    if tab3_delta >= 0:
        st.success("(1) RFC: Threshold is maintained")
    else:
        st.error("(1) RFC: Threshold is not maintained")
    
    stabilityDistance111_112_etc = np.round(LA.norm(shap_values_111_etc[1]-shap_values_112_etc[1]),4)
    tab33_delta = tab3_theta - stabilityDistance111_112_etc
    tab33_col1, tab33_col2 = st.columns(2)
    tab33_col1.metric(label="Euclidean Distance", value=stabilityDistance111_112_etc)
    tab33_col2.metric(label="Threshold Value", value=np.round(tab33_delta,4))
    if tab33_delta >=0:
        st.success("Theta is maintained")
    else:
        st.error("Theta is not maintained")
    st.write("*******************************************************************************************")
    #####################################################################################
    st.subheader("Stability Check for 112 & 113")
    st.write(X_test.loc[[112,113]])
    tmpDF = X_test.loc[[112,113]].copy()
    deltaDF = tmpDF.diff()
    deltaDF = deltaDF.tail(1)
    deltaDF.index.rename("Delta", inplace=True)
    st.write("Delta between feature values of row 112 and row 113")
    st.write(deltaDF)
    instance_113 = X_test.loc[[113]]
    shap_values_113 = explainer.shap_values(instance_113)
    shap_values_113_etc = explainer1.shap_values(instance_113)
    shap_values_112_etc = explainer1.shap_values(instance_112)
    st.write("RFC - SHAP Force Plot for instance 112")
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values_112[1], instance_112))
    st.write("RFC - SHAP Force Plot for instance 113")
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values_113[1], instance_113))
    st.write("ETC - SHAP Force Plot for instance 112")
    st_shap(shap.force_plot(explainer1.expected_value[1], shap_values_112_etc[1], instance_112))
    st.write("ETC - SHAP Force Plot for instance 113")
    st_shap(shap.force_plot(explainer1.expected_value[1], shap_values_113_etc[1], instance_113))
    stabilityDistance112_113 = np.round(LA.norm(shap_values_112[1]-shap_values_113[1]),4)
    tab34_col1, tab34_col2 = st.columns(2)
    
    #tab3 delta
    tab34_delta = tab3_theta-stabilityDistance112_113
    tab34_col1.metric(label="Euclidean Distance", value=stabilityDistance112_113)
    tab34_col2.metric(label="Threshold Value", value=round(tab34_delta,4))
    if tab34_delta >= 0:
        st.success("(1) RFC: Threshold is maintained")
    else:
        st.error("(1) RFC: Threshold is not maintained")
    #tab35_col1, tab35_col2 = st.columns(2)
    #tab35_col1.metric(label="RFC Euclidean Distance", value=1)
    #tab35_col2.metric(label="Threshold Value", value=1)

    stabilityDistance112_113_etc = np.round(LA.norm(shap_values_112_etc[1]-shap_values_113_etc[1]),4)
    tab38_delta = tab3_theta - stabilityDistance112_113_etc
    tab38_col1, tab38_col2 = st.columns(2)
    tab38_col1.metric("ETC Euclidean Distance", value=stabilityDistance112_113_etc)
    tab38_col2.metric("Theta Delta", value=tab38_delta)
    if tab38_delta >= 0:
        st.success("Thetha threshold is maintained")
    else:
        st.error("Thehta threshold is not maintained")
    st.write("*******************************************************************************************")
    #############################################
    st.subheader("Stability Check for instance 68 & 69")
    st.write(X_test.loc[[68,69]])
    tmpDF = X_test.loc[[68,69]].copy()
    deltaDF = tmpDF.diff()
    deltaDF = deltaDF.tail(1)
    deltaDF.index.rename("Delta", inplace=True)
    st.write("Delta between feature values of row 68 and row 69")
    st.write(deltaDF)

    instance_68 = X_test.loc[[68]]
    instance_69 = X_test.loc[[69]]
    #rfc shap values
    shap_values_68 = explainer.shap_values(instance_68)
    shap_values_69 = explainer.shap_values(instance_69)
    #etc shap_values
    shap_values_68_etc = explainer1.shap_values(instance_68)
    shap_values_69_etc = explainer1.shap_values(instance_69)

    st.write("RFC SHAP Force Plot for instance 68")
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values_68[1], instance_68))
    st.write("RFC SHAP Force Plot for instance 69")
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values_69[1], instance_69))
    st.write("ETC SHAP Force Plot for instance 68")
    st_shap(shap.force_plot(explainer1.expected_value[1], shap_values_68_etc[1], instance_68))
    st.write("ETC SHAP Force Plot for instance 69")
    st_shap(shap.force_plot(explainer1.expected_value[1], shap_values_69_etc[1], instance_69))

    tab36_col1, tab36_col2 = st.columns(2)
    stabilityDistance68_69 = np.round(LA.norm(shap_values_68[1]-shap_values_69[1]),4)
    tab36_delta = tab3_theta - stabilityDistance68_69
    tab36_col1.metric("RFC Euclidean distance", value=stabilityDistance68_69)
    tab36_col2.metric("Theta Delta", value=tab36_delta)
    if tab36_delta >= 0:
        st.success("Threshold is maintained")
    else:
        st.error("Threshold is not maintained")

    stabilityDistance68_69_etc = np.round(LA.norm(shap_values_68_etc[1]-shap_values_69_etc[1]),4)
    tab37_delta = tab3_theta - stabilityDistance68_69_etc
    tab37_col1, tab37_col2 = st.columns(2)
    tab37_col1.metric("ETC Euclidean Distance", value=stabilityDistance68_69_etc)
    tab37_col2.metric("Theta Delta", value=tab37_delta)
    if tab37_delta >= 0:
        st.success("Threshold is maintained")
    else:
        st.error("Threshold is not maintained")

    st.write("*******************************************************************************************")
    

with tab4:
    st.subheader("Framework Component - Simplicity")
    expanderComponent4 = st.expander("See explanation")
    expanderComponent4.write("""
    The simplicity component will check a given explanation for its length.
    The assumption for this component is that an explanation with fewer components
    is more explainable compared to an explanation with more components.""")

with tab5:
    st.subheader("Framework Component - Permutation Feature Importance")
    expanderComponent5 = st.expander("See explanation")
    expanderComponent5.write("""The component Permutation Feature importance will analyse the model error
    when every feature column in the data will be permuted. In case the error increases after permuting
    then the underlying model is more dependent on the data. Otherwise if the model error
    is not increasing then the model is not dependent on the data""")

    perm_rfc = PermutationImportance(rfc).fit(X_test, y_test)
    perf_etc = PermutationImportance(etc).fit(X_test, y_test)
    st.subheader("Random Forest Classifier Permutation Feature Importance")
    st.dataframe(eli5.formatters.format_as_dataframe(eli5.explain_weights(perm_rfc, feature_names=X_test.columns.tolist())))
    st.subheader("Extra Trees Classifier Permutation Feature Importance")
    st.dataframe(eli5.formatters.format_as_dataframe(eli5.explain_weights(perf_etc, feature_names=X_test.columns.tolist())))

