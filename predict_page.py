import streamlit as st
import pickle
import numpy as np
import xgboost as xgb

def load_model():
    with open('saved_steps_xgboost.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Extract hyperparameters
xgb_params = regressor.get_params()

def show_predict_page():
    st.title("Data Science Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")

    # Display hyperparameters
    st.write("### XGBoost Hyperparameters")
    st.write(f"Learning Rate: {xgb_params['learning_rate']}")
    st.write(f"Number of Estimators: {xgb_params['n_estimators']}")
    st.write(f"Max Depth: {xgb_params['max_depth']}")
    st.write(f"Subsample: {xgb_params['subsample']}")
    st.write(f"Colsample bytree: {xgb_params['colsample_bytree']}")
    st.write(f"Objective: {xgb_params['objective']}")
    st.write(f"Random State: {xgb_params['random_state']}")

if __name__ == "__main__":
    show_predict_page()
