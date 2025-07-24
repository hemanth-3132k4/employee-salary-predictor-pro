
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from io import StringIO
import base64

# Page settings
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

st.markdown("""
    <style>
        .stApp { background-color: #1a0033; }
        .stButton > button {
            background-color: #d500f9;
            color: black;
            border-radius: 5px;
        }
        html, body, [class*="css"] {
            color: #f3e5f5;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3, h4, h5 { color: #f3e5f5 !important; }
        input, .stSelectbox div span, .stNumberInput input {
            color: #f3e5f5 !important;
            background-color: #33004d;
        }
        svg text { fill: #f3e5f5 !important; }
        .stSlider > div > div > div {
            background-color: #aa00ff !important;
        }
    </style>
""", unsafe_allow_html=True)


 
st.markdown("<h1 style='text-align:center;color:#4A90E2;'>💼 Employee Salary Predictor </h1>", unsafe_allow_html=True)
st.markdown("#### <i>By BAIROLLU HEMANTH KUMAR</i>", unsafe_allow_html=True)

# Help panel
with st.expander("ℹ️ What does this app do?"):
    st.write("""
        This app uses a machine learning model (Random Forest) to predict employee salary based on:
        - Age (18–75)
        - Years of Experience (0–30)
        - Education Level
        - Job Role

        It also shows prediction history, salary vs experience graph, and lets you download results.
    """)

# Input section
st.markdown("### 📋 Enter Employee Details")

age = st.slider("Age", 18, 75, 25)
experience = st.slider("Years of Experience", 0, 30, 1)
education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
job_role = st.selectbox("Job Role", ["Software Engineer", "Data Scientist", "HR", "Manager"])

# Encoding
edu_dict = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
role_dict = {"Software Engineer": 0, "Data Scientist": 1, "HR": 2, "Manager": 3}

# Model training
@st.cache_resource
def train_model():
    data = {
        "Age": [22, 25, 30, 35, 40, 45, 50],
        "Experience": [1, 3, 5, 7, 10, 12, 15],
        "Education": [1, 1, 2, 2, 3, 3, 3],
        "JobRole": [0, 1, 2, 0, 3, 1, 0],
        "Salary": [300000, 450000, 600000, 700000, 1000000, 1200000, 1500000]
    }
    df = pd.DataFrame(data)
    X = df[["Age", "Experience", "Education", "JobRole"]]
    y = df["Salary"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, df

model, df_train = train_model()

# User input
input_data = {
    "Age": age,
    "Experience": experience,
    "Education": edu_dict[education_level],
    "JobRole": role_dict[job_role]
}
input_df = pd.DataFrame([input_data])

# Predict
if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🔮 Predict Salary"):
    predicted_salary = model.predict(input_df)[0]
    input_data["Predicted Salary"] = f"₹ {predicted_salary:,.2f}"
    st.session_state.history.append(input_data)

    # Display result
    st.markdown("### 💰 Prediction Result")
    result_df = pd.DataFrame([input_data])
    result_df["Education"] = education_level
    result_df["JobRole"] = job_role
    st.dataframe(result_df[["Age", "Experience", "Education", "JobRole", "Predicted Salary"]], use_container_width=True)

    # CSV download
    csv = result_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">📥 Download as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

# History log
if st.session_state.history:
    st.markdown("### 📊 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    hist_df["Education"] = hist_df["Education"].map({0:"High School", 1:"Bachelor's", 2:"Master's", 3:"PhD"})
    hist_df["JobRole"] = hist_df["JobRole"].map({0:"Software Engineer", 1:"Data Scientist", 2:"HR", 3:"Manager"})
    st.dataframe(hist_df, use_container_width=True)

# Chart: Salary vs Experience
st.markdown("### 📈 Salary vs. Experience (Sample Data)")
fig = px.line(df_train, x="Experience", y="Salary", color="Education", markers=True)
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<center><i>Made with ❤️ using Streamlit</i></center>", unsafe_allow_html=True)
