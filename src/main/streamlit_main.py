__author__ = "Anil Yelin"
__version__= "0.9"


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
from sklearn.preprocessing import LabelEncoder

with st.sidebar:
    st.subheader("Quick Navigation")
    st.markdown("[Dataset Overview](#dataset-overview)")
    st.markdown("[Model Training](#model-training)")
    st.markdown("[Explainability Section](#explainability-section)")
    st.markdown("[Explainability Checker Framework Architecture](#explainability-checker-framework-architecture)")
    st.markdown("[Explainability Checker Framework](#explainability-checker-framework)")
    st.markdown("[Summary](#summary)")
    st.subheader("Help Section")
    st.write("You can download a manual which explains how to use this Streamlit App")
    with open("/Users/anilyelin/Documents/Masterarbeit/AutomatedXAI/AutomatedXAI/src/manual.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(label="Download Documentation and Manual", data=PDFbyte, file_name="manaul.pdf")

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

def scoreCalculator(score_A, score_B):
    """this function compares the score of the two models and 
        displays the results as streamlit info 
        banner
            input: score_A of model A
                   score B of model B
            return info banner with model
            with higher score"""
    if score_A > score_B:
        st.info("RFC achieved a higher score")
    else:
        st.info("ETC achieved a higher score")
    
st.title("Automated Explainability Checker Framework v0.9")
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

st.title("Explainability Checker Framework Architecture")
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
consistencyTab, robustnessTab, stabilityTab, simplicityTab, permutationTab = st.tabs(["Component Consistency", "Component Robustness", 
"Component Stability","Component Simplicity","Component Feature Importance"])

##### CONSISTENCY COMPONENT ########################################################################################
with consistencyTab:
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
    tableIndex = []
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
                tableIndex.append(indexValue[i])
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

    
    st.subheader("[Consistency] Summary Table")
    st.info("Below you can see a table with all results with respect to your parameters.")
    euclideanDF = pd.DataFrame(tableEuclidean)
    thetaDF = pd.DataFrame(tableTheta)
    indexDF = pd.DataFrame(tableIndex)
    euclideanDF.columns = ["Euclidean Distance SHAP Vectors: RFC <--> ETC"]
    thetaDF.columns = ["Threshold Delta "+str(consistencyThreshold)]
    indexDF.columns = ["Data Instance (index)"]
    df_col_merged = pd.concat([indexDF, euclideanDF, thetaDF], axis=1)
    df_col_merged.index += 1
    st.write(df_col_merged)
    tableFile = convert_df(df_col_merged)
    st.download_button(label="Download results as csv file",data=tableFile, file_name="result_table_consistency.csv")
    st.write("Threshold of: ", consistencyThreshold, " is not maintained for ",int(thetaDF.lt(0).sum()) , " instances of ", kNumber, " instances (in total)")
    consistencyScore = (kNumber-int(thetaDF.lt(0).sum()))/kNumber
    st.write("Consistency Score: ", np.round(consistencyScore*100,2),"%")
        
###### ROBUSTNESS COMPONENT ################################################################################################
with robustnessTab:
    st.subheader("Framework Component - Robustness")
    expander2 = st.expander("See explanation")
    expander2.write("""The component robustness will analyse the explanations for specific data instances
    when marginal changes for some of the features are applied. The key assumption for this component 
    is that for marginal changes in the input data the corresponding explanation after the change should
    deviate also in a marginal way.""")
    robustnessKNumber = st.number_input("Please enter a value for k",min_value=1, max_value=len(X_test)-1, step=1)
    st.write("Entered number k is ",robustnessKNumber)
    robustnessThreshold = st.number_input("Please enter a threshold value", min_value=0.01, max_value=0.5, step=0.01)
    st.write("The following table is showing the k data instances with its corresponding values")
    st.write(X_test.head(robustnessKNumber))
    X_test_copy = X_test.copy(deep=True)
    st.info("Please press the button below or make manual marginal changes")
    if st.button("Apply automated marginal changes"):
        st.write("Calling function for automated marginal changes")
        for i in range(robustnessKNumber):
            automatedChange(X_test, indexValue[i])
        #automatedChange(X_test, 138)
        st.success("Marginal changes applied successfully!")
        st.write("Dataset after marginal changes")
        st.write(X_test.head(robustnessKNumber))
    
    st.subheader("Manual Marginal Changes")
    manualIndex = st.number_input("Please enter an index to modify the data manually", min_value=X_test.index.min(), max_value=X_test.index.max())
    if manualIndex not in X_test.index:
        st.error("The entered index value does not exist. Please try again")
    if manualIndex in X_test.index:
        st.write("Manual Changes for data instance with index: ", manualIndex)
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

    # lists for the summary tables
    tableEuclidean_tab2 = []
    tableTheta_tab2 = []
    tableIndex_tab2 = []
    tableEuclidean_tab2_etc = []
    tableTheta_tab2_etc = []


    st.write("Resulting changes of data instance with index: ")
    
    #this for loop populates the respective data instance with the manually changed values
    deltas = [c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21]
    for i in range(22):
        #X_test.loc[X_test.index[0], [cols[i]]] = [(X_test.iloc[0][cols[i]])+deltas[i]]
        X_test.loc[manualIndex, [cols[i]]] = [(X_test.loc[manualIndex][cols[i]])+deltas[i]]
    #X_test.loc[X_test.index[0], [cols[1]]] = [(X_test.iloc[0][cols[1]])+c1]
    #X_test.loc[X_test.index[0], [cols[2]]] = [(X_test.iloc[2][cols[2]])+c2]
    #X_test.loc[X_test.index[0], [cols[3]]] = [(X_test.iloc[3][cols[3]])+c3]
    st.write(X_test.loc[[manualIndex]])
    #st.write(X_test.head(robustnessKNumber))
    ##### NEW FOR LOOP
    for i in range(robustnessKNumber):
        #copy
        st.subheader("Random Forest Classifier Results")
        instance_robustness_copy = X_test_copy.loc[[indexValue[i]]]
        shap_values_robustness_copy = explainer.shap_values(instance_robustness_copy)
        #marginal change
        instance_robustness = X_test.loc[[indexValue[i]]]
        shap_values_robustness = explainer.shap_values(instance_robustness)
        st.write("Orignal Explanation for instance: ", indexValue[i])
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values_robustness_copy[1], instance_robustness_copy))
        st.write("Explanation after marginal changes for instance: ", indexValue[i])
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values_robustness[1], instance_robustness))

        #calculation of euclidean distance
        robustnessEuclideanDistance = np.round(LA.norm(shap_values_robustness_copy[1]-shap_values_robustness[1]),4)
        tableEuclidean_tab2.append(robustnessEuclideanDistance)
        #calculation of threshold difference
        robustnessThresholdDifference = robustnessThreshold - robustnessEuclideanDistance
        tableTheta_tab2.append(robustnessThresholdDifference)

        tableIndex_tab2.append(indexValue[i])
        tab2_col1, tab2_col2 = st.columns(2)
        tab2_col1.metric("Euclidean Distance",robustnessEuclideanDistance)
        tab2_col2.metric("Delta Value", np.round(robustnessThresholdDifference,4))
        if robustnessThresholdDifference >=0:
            st.success("[RFC] Threshold Difference is maintained")
        else:
            st.error("[RFC] Threshold Difference is not maintained")
        st.subheader("Extra Trees Classifier Results")
        
        etc_shap_values_robustness_copy = explainer1.shap_values(instance_robustness_copy)
        #marginal changes
        etc_shap_values_robustness = explainer1.shap_values(instance_robustness)
        st.write("Orignal Explanation for instance: ", indexValue[i])
        st_shap(shap.force_plot(explainer1.expected_value[1], etc_shap_values_robustness_copy[1], instance_robustness_copy))
        st.write("Explanation after marginal changes for instance: ", indexValue[i])
        st_shap(shap.force_plot(explainer1.expected_value[1], etc_shap_values_robustness[1], instance_robustness))
        #euclidean distance
        etc_robustnessEuclideanDistance = np.round(LA.norm(etc_shap_values_robustness[1]-etc_shap_values_robustness_copy[1]),4)
        tableEuclidean_tab2_etc.append(etc_robustnessEuclideanDistance)
        etc_robustnessThresholdDifference = robustnessThreshold-etc_robustnessEuclideanDistance
        tableTheta_tab2_etc.append(etc_robustnessThresholdDifference)
        tab22_col1, tab22_col2 = st.columns(2)
        tab22_col1.metric("Euclidean Distance", etc_robustnessEuclideanDistance)
        tab22_col2.metric("Delta Values", np.round(etc_robustnessThresholdDifference,4))
        if etc_robustnessThresholdDifference >= 0:
            st.success("[ETC] Threshold is maintained")
        else:
            st.error("[ETC] Threshold is not maintained")
        st.write("*******************************************************************************************")
   
    st.subheader("[Robustness] Summary Table")
        #st.write("Below you can find a table with all results")
    tab2_euclideanDF = pd.DataFrame(tableEuclidean_tab2)
    tab2_thetaDF = pd.DataFrame(tableTheta_tab2)
    tab2_indexDF = pd.DataFrame(tableIndex_tab2)
    tab2_euclideanDF_etc = pd.DataFrame(tableEuclidean_tab2_etc)
    tab2_thetaDF_etc = pd.DataFrame(tableTheta_tab2_etc)
    tab2_euclideanDF.columns = ["Euclidean Distance (RFC)"]
    tab2_thetaDF.columns = ["Threshold Delta (RFC) "+str(robustnessThreshold)]
    tab2_euclideanDF_etc.columns = ["Euclidean Distance (ETC)"]
    tab2_thetaDF_etc.columns = ["Threshold Delta (ETC)"]
    tab2_indexDF.columns = ["Index Value"]
    tab2_df_col_merged = pd.concat([tab2_indexDF, tab2_euclideanDF, tab2_thetaDF, tab2_euclideanDF_etc, tab2_thetaDF_etc], axis=1)
    tab2_df_col_merged.index += 1
    st.write(tab2_df_col_merged)
    tableFile_robustness = convert_df(tab2_df_col_merged)
    st.download_button(label="Download results as csv file",data=tableFile_robustness, file_name="result_table_robustness.csv")
    st.subheader("RFC")
    st.write("Threshold of ", robustnessThreshold," is not maintained for ", int(tab2_thetaDF.lt(0).sum()), "instances of ", robustnessKNumber, " instances in total")
    st.subheader("ETC")
    st.write("Threshold of ", robustnessThreshold," is not maintained for ", int(tab2_thetaDF_etc.lt(0).sum()), "instances of ", robustnessKNumber, " instances in total")
    robustnessScore_rfc = int(tab2_thetaDF.lt(0).sum())
    robustnessScore_etc = int(tab2_thetaDF_etc.lt(0).sum())
    if robustnessScore_rfc < robustnessScore_etc:
        st.success("RFC model has a better score in terms of robustness")
    else:
        st.success("ETC model has a better score in terms of robustness")

##### STABILITY COMPONENT #####################################################################################################
with stabilityTab:
    st.subheader("Framework Component - Stability")
    expanderComponent3 = st.expander("See explanation")
    expanderComponent3.write("""The component stability will analyse the explanations of 
    neighboring data points. The key assumption is that for neighboring data points the 
    explanations should also similar""")
    st.info("Three neighboring data instances has been found in the Parkinson dataset")
    tab3_kValue = st.number_input("Please enter the value k for neighboring data instances", min_value=1, max_value=3, value=3, step=1, disabled=True)
    tab3_theta = st.number_input("Please enter the threshold value theta", min_value=0.01, max_value=0.5, step=0.01)
    #tables
    tab3_euclidean_distance_rfc = []
    tab3_euclidean_distance_etc = []
    tab3_theta_rfc = []
    tab3_theta_etc = []
    tab3_index = ["111 <-> 112","112 <-> 113", "68 <-> 69"]
    # end tables
    st.subheader("Stability Check for 111 & 112")
    st.write(X_test.loc[[111,112]])
    tmpDF = X_test.loc[[111,112]].copy()
    deltaDF = tmpDF.diff()
    deltaDF = deltaDF.tail(1)
    deltaDF.index.rename("Delta", inplace=True)
    st.info("Below you can see the delta values between instance 111 and 112")
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
    tab3_euclidean_distance_rfc.append(stabilityDistance)
    #tab3 delta
    tab3_delta = tab3_theta-stabilityDistance
    tab3_theta_rfc.append(tab3_delta)
    tab3_col1.metric(label="RFC Euclidean Distance", value=stabilityDistance)
    tab3_col2.metric(label="Threshold Value", value=round(tab3_delta,4))
    if tab3_delta >= 0:
        st.success("RFC threshold is maintained")
    else:
        st.error("RFC threshold is not maintained")
    
    stabilityDistance111_112_etc = np.round(LA.norm(shap_values_111_etc[1]-shap_values_112_etc[1]),4)
    tab3_euclidean_distance_etc.append(stabilityDistance111_112_etc)
    tab33_delta = tab3_theta - stabilityDistance111_112_etc
    tab3_theta_etc.append(tab33_delta)
    tab33_col1, tab33_col2 = st.columns(2)
    tab33_col1.metric(label="ETC Euclidean Distance", value=stabilityDistance111_112_etc)
    tab33_col2.metric(label="Threshold Value", value=np.round(tab33_delta,4))
    if tab33_delta >=0:
        st.success("ETC threshold is maintained")
    else:
        st.error("ETC threshold is not maintained")
    st.write("*******************************************************************************************")
    #####################################################################################
    st.subheader("Stability Check for 112 & 113")
    st.write(X_test.loc[[112,113]])
    tmpDF = X_test.loc[[112,113]].copy()
    deltaDF = tmpDF.diff()
    deltaDF = deltaDF.tail(1)
    deltaDF.index.rename("Delta", inplace=True)
    st.info("Below you can see the delta values between instance 112 and 113")
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
    tab3_euclidean_distance_rfc.append(stabilityDistance112_113)
    tab34_col1, tab34_col2 = st.columns(2)
    
    #tab3 delta
    tab34_delta = tab3_theta-stabilityDistance112_113
    tab3_theta_rfc.append(tab34_delta)
    tab34_col1.metric(label="RFC Euclidean Distance", value=stabilityDistance112_113)
    tab34_col2.metric(label="Threshold Value", value=round(tab34_delta,4))
    if tab34_delta >= 0:
        st.success("RFC: Threshold is maintained")
    else:
        st.error("RFC: Threshold is not maintained")
    #tab35_col1, tab35_col2 = st.columns(2)
    #tab35_col1.metric(label="RFC Euclidean Distance", value=1)
    #tab35_col2.metric(label="Threshold Value", value=1)

    stabilityDistance112_113_etc = np.round(LA.norm(shap_values_112_etc[1]-shap_values_113_etc[1]),4)
    tab3_euclidean_distance_etc.append(stabilityDistance112_113_etc)
    tab38_delta = tab3_theta - stabilityDistance112_113_etc
    tab3_theta_etc.append(tab38_delta)
    tab38_col1, tab38_col2 = st.columns(2)
    tab38_col1.metric("ETC Euclidean Distance", value=stabilityDistance112_113_etc)
    tab38_col2.metric("Theta Delta", value=np.round(tab38_delta,4))
    if tab38_delta >= 0:
        st.success("ETC threshold is maintained")
    else:
        st.error("ETC threshold is not maintained")
    st.write("*******************************************************************************************")
    #############################################
    st.subheader("Stability Check for instance 68 & 69")
    st.write(X_test.loc[[68,69]])
    tmpDF = X_test.loc[[68,69]].copy()
    deltaDF = tmpDF.diff()
    deltaDF = deltaDF.tail(1)
    deltaDF.index.rename("Delta", inplace=True)
    st.info("Below you can see the delta values between instance 68 and 69")
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
    tab3_euclidean_distance_rfc.append(stabilityDistance68_69)
    tab36_delta = tab3_theta - stabilityDistance68_69
    tab3_theta_rfc.append(tab36_delta)
    tab36_col1.metric("RFC Euclidean distance", value=stabilityDistance68_69)
    tab36_col2.metric("Theta Delta", value=np.round(tab36_delta,4))
    if tab36_delta >= 0:
        st.success("RFC threshold is maintained")
    else:
        st.error("RFC threshold is not maintained")

    stabilityDistance68_69_etc = np.round(LA.norm(shap_values_68_etc[1]-shap_values_69_etc[1]),4)
    tab3_euclidean_distance_etc.append(stabilityDistance68_69)

    tab37_delta = tab3_theta - stabilityDistance68_69_etc
    tab3_theta_etc.append(tab37_delta)
    tab37_col1, tab37_col2 = st.columns(2)
    tab37_col1.metric("ETC Euclidean Distance", value=stabilityDistance68_69_etc)
    tab37_col2.metric("Theta Delta", value=np.round(tab37_delta,4))
    if tab37_delta >= 0:
        st.success("ETC threshold is maintained")
    else:
        st.error("ETC threshold is not maintained")

    st.write("*******************************************************************************************")

    st.subheader("[Stability] Summary Table")
    tab3_euclideanDF_rfc = pd.DataFrame(tab3_euclidean_distance_rfc)
    tab3_thetaDF_rfc = pd.DataFrame(tab3_theta_rfc)
    tab3_euclideanDF_etc = pd.DataFrame(tab3_euclidean_distance_etc)
    tab3_thetaDF_etc = pd.DataFrame(tab3_theta_etc)
    tab3_indexDF = pd.DataFrame(tab3_index)
    tab3_indexDF.columns = ["Neigboring Data Instances"]
    tab3_euclideanDF_rfc.columns = ["Euclidean Distance RFC"]
    tab3_euclideanDF_etc.columns = ["Euclidean Distance ETC"]
    tab3_thetaDF_rfc.columns = ["Theta RFC"]
    tab3_thetaDF_etc.columns = ["Theta ETC"]
    tab3_table_merge = pd.concat([tab3_indexDF, tab3_euclideanDF_rfc, tab3_thetaDF_rfc, tab3_euclideanDF_etc,tab3_thetaDF_etc], axis=1)
    tab3_table_merge.index += 1
    st.write(tab3_table_merge)
    tableFile_stability = convert_df(tab3_table_merge)
    st.download_button(label="Download results as csv file",data=tableFile_stability, file_name="result_table_stability.csv")
    st.subheader("RFC")
    st.write("Threshold of ", tab3_theta," is not maintained for ", int(tab3_thetaDF_rfc.lt(0).sum()), "instances of 3 instances in total")

    st.subheader("ETC")
    st.write("Threshold of ", tab3_theta," is not maintained for ", int(tab3_thetaDF_etc.lt(0).sum()), "instances of 3 instances in total")
    rfc_stabilityScore = int(tab3_thetaDF_rfc.lt(0).sum())
    etc_stabilityScore = int(tab3_thetaDF_etc.lt(0).sum())
    if rfc_stabilityScore < etc_stabilityScore:
        st.success("RFC model has a better score in terms of stability component")
    else:
        st.success("ETC model has a better score in terms of stability component")
    
##### SIMPLICITY COMPONENT ##################################################################################
def calcNegativeSHAPScoreRFC(indexVal):
    instance = X_test.loc[[indexVal]]
    shap_values = explainer.shap_values(instance)
    #convert to dataframe
    df = pd.DataFrame({"Score":shap_values[1][0,:]}) 
    return df.lt(0).sum()

def calcNegativeSHAPScoreETC(indexVal):
    instance = X_test.loc[[indexVal]]
    shap_values = explainer1.shap_values(instance)
    #convert to dataframe
    df = pd.DataFrame({"Score":shap_values[1][0,:]}) 
    return df.lt(0).sum()

with simplicityTab:
    st.subheader("Framework Component - Simplicity")
    expanderComponent4 = st.expander("See explanation")
    expanderComponent4.write("""
    The simplicity component will check a given explanation for its length.
    The assumption for this component is that an explanation with fewer components
    is more explainable compared to an explanation with more components.""")
    simplicityKNumber = st.number_input("Please enter a value for parameter k", min_value=1, max_value=len(X_test))
    st.write("Parameter k is: ", simplicityKNumber)
    st.write(X_test.head(simplicityKNumber))
    st.info("Calculating for each of the k data instances the non negative SHAP scores")
    simplicity_tab_rfc = []
    simplicity_tab_etc = []
    for i in range(simplicityKNumber):
        st.subheader("RFC")
        res_rfc = calcNegativeSHAPScoreRFC(indexValue[i])
        st.write("The number of negative SHAP scores for instance: ", indexValue[i], " is:", int(res_rfc), " of ", X_test.iloc[0].shape[0]," components")
        simplicity_tab_rfc.append(int(res_rfc))
        st.subheader("ETC")
        res_etc = calcNegativeSHAPScoreETC(indexValue[i])
        simplicity_tab_etc.append(int(res_etc))
        st.write("The number of negative SHAP score for instance: ", indexValue[i], " is: ", int(res_etc), " of ", X_test.iloc[0].shape[0]," components")
        st.write("*************************************************************************************")

    st.subheader("[Simplicity] Summary Table")
    st.info("Below you can find the summary table for the component simplicity")
    simplicity_tab_rfc_df = pd.DataFrame(simplicity_tab_rfc)
    simplicity_tab_etc_df = pd.DataFrame(simplicity_tab_etc)
    simplicity_tab_rfc_df.columns = ["[RFC] Negative SHAP scores"]
    simplicity_tab_etc_df.columns = ["[ETC] Negative SHAP scores"]
    simplicity_tab_merged = pd.concat([simplicity_tab_rfc_df, simplicity_tab_etc_df],axis=1)
    st.write(simplicity_tab_merged)
    tableFile_simplicity = convert_df(simplicity_tab_merged)
    rfc_sum = simplicity_tab_rfc_df["[RFC] Negative SHAP scores"].sum()
    etc_sum = simplicity_tab_etc_df["[ETC] Negative SHAP scores"].sum()
    denominator = simplicityKNumber*22
    rfc_final_score = rfc_sum / denominator
    etc_final_score = etc_sum / denominator
    st.write("Sum of RFC: ", np.round(rfc_final_score*100, 4), "%")
    st.write("Sum of ETC: ",np.round(etc_final_score*100,4), "%")
    if rfc_final_score < etc_final_score:
        st.success("RFC model has a better scoring in terms of the simplicity component")
    else:
        st.success("ETC model has a better scoring in terms of the simplicity component")
    st.download_button(label="Download results as csv file",data=tableFile_simplicity, file_name="result_table_simplicity.csv")
###### PERMUTATION FEATURE IMPORTANCE COMPONENT ############################################################

with permutationTab:
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
    rfc_tmp_df = eli5.formatters.format_as_dataframe(eli5.explain_weights(perm_rfc, feature_names=X_test.columns.tolist()))
    tableFile_eli5_rfc = convert_df(rfc_tmp_df)
    st.download_button(label="Download results as csv file",data=tableFile_eli5_rfc, file_name="result_table_eli5_rfc.csv")
    rfc_eli5_sum = rfc_tmp_df['weight'].sum()
    
    st.write("Total Weight: ", np.round(rfc_eli5_sum,4))
   
    st.subheader("Extra Trees Classifier Permutation Feature Importance")
    st.dataframe(eli5.formatters.format_as_dataframe(eli5.explain_weights(perf_etc, feature_names=X_test.columns.tolist())))
    etc_tmp_df = eli5.formatters.format_as_dataframe(eli5.explain_weights(perf_etc, feature_names=X_test.columns.tolist()))
    tableFile_eli5_etc = convert_df(etc_tmp_df)
    st.download_button(label="Download results as csv file",data=tableFile_eli5_etc, file_name="result_table_eli5_etc.csv")
    etc_eli5_sum = etc_tmp_df['weight'].sum()
    st.write("Total Weight: ", np.round(etc_eli5_sum,4))

    if rfc_eli5_sum >= etc_eli5_sum:
        st.write("Conclusion: Black Box Model <RFC> has a higher total of weight of: ", np.round(rfc_eli5_sum,4))
    else:
        st.write("Conclusion: Black Box Model <ETC> has a higher total of weight of: ", np.round(etc_eli5_sum,4))



st.header("Summary")

st.info("Below you can find all relevant table at one place")

st.subheader("Conistency Check")

st.write(df_col_merged)

st.subheader("Robustness Check")

st.write(tab2_df_col_merged)

if robustnessScore_rfc < robustnessScore_etc:
    st.success("RFC model has a better score in terms of robustness")
else:
        
    st.success("ETC model has a better score in terms of robustness")

st.subheader("Stability Check")

st.write(tab3_table_merge)

if rfc_stabilityScore < etc_stabilityScore:
    st.success("RFC model has a better score in terms of stability component")
else:
    st.success("ETC model has a better score in terms of stability component")

st.subheader("Simplicity Check")

st.write(simplicity_tab_merged)

if rfc_final_score < etc_final_score:
    st.success("RFC model has a better scoring in terms of the simplicity component")
else:
    st.success("ETC model has a better scoring in terms of the simplicity component")

st.subheader("Permutation Feature Importance")
if rfc_eli5_sum >= etc_eli5_sum:
    st.write("Conclusion: Black Box Model <RFC> has a higher total of weight of: ", np.round(rfc_eli5_sum,4))
else:
    st.write("Conclusion: Black Box Model <ETC> has a higher total of weight of: ", np.round(etc_eli5_sum,4))

