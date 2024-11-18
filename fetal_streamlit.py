# App to predict the chances of admission using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import sklean
import pickle
import warnings
warnings.filterwarnings('ignore')

st.cache_data

# Set up the app title and image
st.title('Fetal Health Classification: A Machine Learning App') #change to diamond emoji
st.image('fetal_health_image.gif', use_column_width = True)
st.write("Utilize our advanced Machine Learning application to predict fetal health classification.") 


# Load the default dataset
default_df = pd.read_csv('fetal_health.csv')

# side bar stuffs
st.sidebar.subheader('Fetal Health Features Input')
user_fetal = st.sidebar.file_uploader('Upload your data', help='File must be in CSV')
st.sidebar.warning('Ensure your data strictly follows the format outlined below.', icon="⚠️" )
st.sidebar.dataframe(default_df.head()) 



user_ml = st.sidebar.radio("Choose Model for Prediction",options= ['Random Forest', 'Decision Tree', 'Adaboost', 'Soft Voting'])
st.sidebar.info(f'You selected: {user_ml}', icon='✔️')



#if option 1 chosen
if user_fetal is None:

    st.info('Please upload data to proceed.', icon="ℹ️")

    pass

if user_fetal is not None:

    st.success('*CSV file uploaded successfully.*', icon="✅")

    st.write(f'### Predicting Fetal Health Class Using {user_ml} Model')

    # read file
    user_df = pd.read_csv(user_fetal)  

    user_len = len(user_df)
    
    #Encode the inputs for model prediction
    encode_df = default_df.copy()
    encode_df = encode_df.drop(columns=['fetal_health'])


    # Combine the list of user data as a row to default_df
    encode_df = pd.concat([encode_df, user_df], ignore_index=True)

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(user_len)

    if user_ml == 'Random Forest':
        rf_pickle = open('fetal_rf.pickle', 'rb') 
        rf_model = pickle.load(rf_pickle) 
        rf_pickle.close()

        # Get the prediction with its intervals
        prediction = rf_model.predict(user_encoded_df)
        pred_prob = rf_model.predict_proba(user_encoded_df)


    if user_ml == 'Decision Tree':
        dt_pickle = open('fetal_dt.pickle', 'rb') 
        dt_model = pickle.load(dt_pickle) 
        dt_pickle.close()

        # Get the prediction with its intervals
        prediction = dt_model.predict(user_encoded_df)
        pred_prob = dt_model.predict_proba(user_encoded_df)
 
    
    if user_ml == 'Adaboost':
        ada_pickle = open('fetal_ada.pickle', 'rb') 
        ada_model = pickle.load(ada_pickle) 
        ada_pickle.close()

        # Get the prediction with its intervals
        prediction = ada_model.predict(user_encoded_df)
        pred_prob = ada_model.predict_proba(user_encoded_df)


    if user_ml == 'Soft Voting':
        sv_pickle = open('fetal_sv.pickle', 'rb') 
        sv_model = pickle.load(sv_pickle) 
        sv_pickle.close()

        # Get the prediction with its intervals
        prediction = sv_model.predict(user_encoded_df)
        pred_prob = sv_model.predict_proba(user_encoded_df)



    user_pred_df = user_df
    pred_prob_rows = pred_prob.max(axis=1) * 100
    user_pred_df['Predicted Class'] = prediction
    user_pred_df['Prediction Probability (%)'] = pred_prob_rows

    user_pred_df['Prediction Probability (%)'] = user_pred_df['Prediction Probability (%)'].apply(lambda x: '{:,.1f}'.format(x))

    
    #Add color to each class
    def color_cells(val):

        if val == 'Normal':
            color = 'lime'
        elif val == 'Suspect':
            color = 'yellow'
        elif val == 'Pathological':
            color = 'orange'
        else:
            color = 'black'
        
        return f'background-color: {color}'

    st.dataframe(user_pred_df.style.applymap(color_cells))


    # Additional tabs for DT model performance
    st.subheader("Model Insights")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix",
                                "Classification Report",
                                "Feature Importance"])

    if user_ml == 'Random Forest':
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_rf.svg')
            st.caption("Confusion Matrix of model predictions.")    

        with tab2:
            class_report_rf = pd.read_csv('class_report_rf.csv', index_col=0)
            class_report_rf = class_report_rf.style.format("{:.2f}")
            #asked ChatGpt to reformat code below to after styler errors
                #columns = class_report_rf.columns.to_list()
                #styled_rf = class_report_rf.background_gradient(subset= columns, cmap='Greens')

            for col in class_report_rf.columns:
                styled_rf = class_report_rf.background_gradient(subset=[col], cmap='Greens')

            st.write("### Classification")
            st.write(styled_rf)
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health condition")   

        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_rf.svg')
            st.caption("features used in this prediction are ranked by relative importance.")

    if user_ml == 'Decision Tree':
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_dt.svg')
            st.caption("Confusion Matrix of model predictions.")    

        with tab2:
            class_report_dt = pd.read_csv('class_report_dt.csv', index_col=0)
            class_report_dt = class_report_dt.style.format("{:.2f}")       
            
            for col in class_report_dt.columns:
                styled_dt = class_report_dt.background_gradient(subset=[col], cmap='Blues')

            st.write("### Classification")
            st.write(styled_dt)
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health condition")   

        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_dt.svg')
            st.caption("features used in this prediction are ranked by relative importance.") 

    if user_ml == 'Adaboost':
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_ada.svg')
            st.caption("Confusion Matrix of model predictions.")    

        with tab2:
            class_report_ada = pd.read_csv('class_report_ada.csv', index_col=0)
            class_report_ada = class_report_ada.style.format("{:.2f}")
            for col in class_report_ada.columns:
                styled_ada = class_report_ada.background_gradient(subset=[col], cmap='Reds')

            st.write("### Classification")
            st.write(styled_ada)
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health condition")   

        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_ada.svg')
            st.caption("features used in this prediction are ranked by relative importance.")  

    if user_ml == 'Soft Voting':
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_sv.svg')
            st.caption("Confusion Matrix of model predictions.")    

        with tab2:
            class_report_sv = pd.read_csv('class_report_sv.csv', index_col=0)
            class_report_sv = class_report_sv.style.format("{:.2f}")
            for col in class_report_sv.columns:
                styled_sv = class_report_sv.background_gradient(subset=[col], cmap='PuOr')

            st.write("### Classification")
            st.write(styled_sv)
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health condition")   

        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_sv.svg')
            st.caption("features used in this prediction are ranked by relative importance.")   
