import streamlit as st
import pandas as pd
import joblib

st.title("🎓 Student CGPA Prediction")

# Load model
model = joblib.load("../model/cgpa_prediction_model.pkl")

st.write("Enter student details below:")

attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
assignment_marks = st.number_input("Assignment Marks", min_value=0.0, max_value=100.0)
reading_time = st.number_input("Reading Time (hrs/day)", min_value=0.0)
writing_time = st.number_input("Writing Time (hrs/day)", min_value=0.0)
avg_marks = st.number_input("Average Subject Marks", min_value=0.0, max_value=100.0)

if st.button("Predict CGPA"):
    input_df = pd.DataFrame({
        'Attendance_%': [attendance],
        'Assignment_Marks': [assignment_marks],
        'Reading_Time_hrs': [reading_time],
        'Writing_Time_hrs': [writing_time],
        'Avg_Subject_Marks': [avg_marks]
    })

    st.write("🔍 Input Data:")
    st.dataframe(input_df)

    prediction = model.predict(input_df)

    st.success(f"📘 Predicted CGPA: {prediction[0]:.2f}")