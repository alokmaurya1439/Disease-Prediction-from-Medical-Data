import pickle

import streamlit as st
import pandas as pd

def load_model():
    with open("best_model.pkl", "rb")as file:
        return pickle.load(file)
    
model = load_model()

st.set_page_config(page_title="Disease Prediction From Medical Data", page_icon="🩺", layout="wide")


st.title("Disease Prediction From Medical Data")
st.markdown(
    "Enter patient heart-health measurements and "
    "review a fast model-based risk estimate.")

with st.sidebar:
    st.header("About")
    st.info("""
        This application predicts the risk of heart disease using machine learning. 
        🔍 Model: Random Forest / SVM / XGBoost  
        📊 Accuracy: ~84%  
        🧠 Input: Patient medical data  
        🎯 Output: Risk level (Low / High)

        ⚠️ Note: This is not a medical diagnosis tool. Consult a doctor for real advice.
        """)
    

    
    st.write(
        "**Features**:" 
        "Age, Sex, Chest Pain Type, Max Heart Rate, Exercise Induced Angina, ST depression, Slope, CA (number of vessels), Thal" 
       
    )
    st.markdown("---")

    st.subheader("Units")
    st.sidebar.markdown("""
        - **Age** → Years  
        - **Max Heart Rate** → bpm  
        - **ST Depression (Oldpeak)** → Numeric value  
        - **CA (Blocked Vessels)** → 0–3  
        - **Thal** → Blood disorder type  
        - **Chest Pain Type** → 4 categories  
        - **Resting ECG** → 3 categories
        - **Slope** → 3 categories
        """)
    

cp_dict = {"Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 3,}

thal_dict = {
        "Unknown / not recorded": 0,
        "Normal": 1,
        "Fixed defect": 2,
        "Reversible defect": 3,
        }


slope_dict = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2,
        }



st.header("Patient Measurements")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("**Age**", min_value=1, max_value=100)
    sex = st.radio("**Gender**", ["Female", "Male"], horizontal=True)
    cp = st.selectbox("**Chest Pain**", 
       ["Select"] + list(cp_dict.keys()), index=0)
       
    

with col2:
    thal = st.selectbox("**Blood flow defect**", ["Select"] + list(thal_dict.keys()), index=0)
    
    thalach = st.number_input(
                "**Heart performance**", min_value=0, max_value=230, value=60
            )
    exang = st.radio("**Exercise chest pain**", ["No", "Yes"], horizontal=True)

with col3:
   
    oldpeak = st.number_input(
                "**Heart stress**", min_value=0.0, max_value=7.0, value=1.0, step=0.1
            )
    slope = st.selectbox("**Heart Rate Measurement Report**", ["Select"] + list(slope_dict.keys()), index=0)
    ca = st.slider("**Vessel blockage**", 0, 4, 0)
    
st.header("🔍Prediction Result")
if st.button("Analyze Risk"):
    missing_selection = (
        cp == "Select"
        or thal == "Select"
        or slope == "Select"
        )

    if missing_selection:
        st.error("Please select all required fields.")


    else:
        feature_order = list(model.feature_names_in_)
        input_data = pd.DataFrame(
            {
                "age": [age],
                "sex": [1 if sex == "Male" else 0],
                "cp": [cp_dict[cp]],
                "thalach": [thalach],
                "exang": [1 if exang == "Yes" else 0],
                "oldpeak": [float(oldpeak)],
                "slope": [slope_dict[slope]],
                "ca": [ca],
                "thal": [thal_dict[thal]]
            }
        )[feature_order]

        prediction =model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction == 1:
            st.write("High Risk of Heart Disease ⚠️")

        else:
            st.write("Low Risk / No Disease ✅")

        st.subheader("Prediction Probability")
        st.write(
            f"Probability of Heart Disease: {prediction_proba[0][1]:.2%}"
        )
        st.write(
            f"Probability of No Heart Disease: {prediction_proba[0][0]:.2%}"
        )