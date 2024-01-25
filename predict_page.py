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

    country = st.sidebar.selectbox("Country", countries)
    education = st.sidebar.selectbox("Education Level", education)

    expericence = st.sidebar.slider("Years of Experience", 0, 50, 3)

    ok = st.sidebar.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.sidebar.subheader(f"The estimated salary is ${salary[0]:.2f}")

    # Display hyperparameters in a dropdown on the left sidebar
    st.sidebar.write("### XGBoost Hyperparameters")
xgb_params['learning_rate'] = st.sidebar.selectbox("Learning Rate", [0.01, 0.1, 0.2], index=1)
xgb_params['n_estimators'] = st.sidebar.selectbox("Number of Estimators", [50, 100, 200], index=1)
xgb_params['max_depth'] = st.sidebar.selectbox("Max Depth", [3, 5, 7], index=1)
xgb_params['subsample'] = st.sidebar.selectbox("Subsample", [0.6, 0.8, 1.0], index=1)
xgb_params['colsample_bytree'] = st.sidebar.selectbox("Colsample bytree", [0.6, 0.8, 1.0], index=1)
xgb_params['objective'] = st.sidebar.selectbox("Objective", ["reg:squarederror", "reg:squaredlogerror"], index=0)
xgb_params['random_state'] = st.sidebar.selectbox("Random State", [42, 123, 456], index=0)


if __name__ == "__main__":
    show_predict_page()
