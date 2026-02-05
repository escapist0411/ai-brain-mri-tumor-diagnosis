import streamlit as st

st.set_page_config(layout="centered")

st.title("👤 Patient Registration")

with st.form("patient_form"):
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 0, 120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    symptoms = st.text_area("Symptoms / Notes")

    submit = st.form_submit_button("Save Patient Details")

if submit:
    st.session_state["patient"] = {
        "name": name,
        "age": age,
        "gender": gender,
        "symptoms": symptoms
    }
    st.success("✅ Patient details saved. Go to MRI Diagnosis.")
