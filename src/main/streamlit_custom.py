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



st.title("Automated Explainability Checker Framework v1.0 - for custom datasets")
st.text("This streamlit app is a prototype for the proposed explainability framework\n"
"proposed in my master thesis")

#### DATASET SECTION #####################################################################

st.header("Dataset Overview")
st.info("""You can upload your own dataset to this prototyp. Please note that the 
    dataset has to be a csv file""")
csvFile = st.file_uploader("Choose a .csv file", accept_multiple_files=False)
if csvFile is not None:
    st.success("File loaded successfully")
    customDF = pd.read_csv(csvFile)
    st.write(customDF)
    labelEncoder = LabelEncoder()
    customDF['variety'] = labelEncoder.fit_transform(customDF['variety'])
    st.info("After applying Label Encoding to non numerical columns")
    st.write(customDF)
    st.info("Please select the column which is the target for the model training")
    targetColumn = st.selectbox("Select the target", customDF.columns)
    st.write("The selected target is", targetColumn)

else:
    st.error("An error occured while loading the file")


### MODEL TRAINING SECTION ########################################

st.header("Model Training")
randomForest_tab, extraTrees_tab = st.tabs(["Random Forest Classifier", "Extra Trees Classifier"])


with randomForest_tab:
    blackBoxModels = ["Random Forest Classifier","Extra Trees Classifier"]
    modelChoice = st.selectbox("Please choose the first black box model for training",blackBoxModels)
    st.subheader("Random Forest Classifier")
    n_estimators = st.number_input("Please enter the number of estimators", min_value=10, step=1, value=50)
    random_state = st.number_input("Please enter a random state number", min_value=0, step=1, value=1)
    test_size = st.number_input("Please enter the size of the test dataset", min_value=0.1, max_value=0.4, value=0.2)
    max_depth = st.number_input("Please enter the max depth for a tree", min_value=10, value=10)
    st.caption("Hyperparameter Summary")
    data = customDF
    #target = data['variety']
    target = data[targetColumn]
    features = data.drop(columns=['variety'])
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
    data = customDF
    #target = data['variety']
    target = data[targetColumn]
    features = data.drop(columns=['variety'])
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


### EXPLAINABILITY SECTION ##########################################

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
consistencyTab, robustnessTab, simplicityTab, permutationTab = st.tabs(["Component Consistency", "Component Robustness"
,"Component Simplicity","Component Feature Importance"])

##### CONSISTENCY COMPONENT ########################################################################################

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
    


    # lists for the summary tables
    tableEuclidean_tab2 = []
    tableTheta_tab2 = []
    tableIndex_tab2 = []
    tableEuclidean_tab2_etc = []
    tableTheta_tab2_etc = []


    st.write("Resulting changes of data instance with index: ")
    

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
