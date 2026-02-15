import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load Models and Preprocessor
# --------------------------------------------------
model = joblib.load("best_model.pkl")
rul_model = joblib.load("rul_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

st.title("Predictive Maintenance System")
st.write("AI-driven industrial machine health monitoring and failure prediction.")

# --------------------------------------------------
# Sidebar Inputs (All Dataset Features)
# --------------------------------------------------
st.sidebar.header("Machine Inputs")

machine_model = st.sidebar.selectbox(
    "Machine Model", 
    ["Model_A", "Model_B", "Model_C"]
)

temp = st.sidebar.number_input("Average Temperature (°C)", 0.0, 150.0, 50.0)
vibration = st.sidebar.number_input("Vibration Level", 0.0, 20.0, 2.0)
rot_speed = st.sidebar.number_input("Rotating Speed (RPM)", 0.0, 6000.0, 1500.0)
voltage = st.sidebar.number_input("Voltage Fluctuation", 0.0, 300.0, 5.0)
torque = st.sidebar.number_input("Torque (Nm)", 0.0, 200.0, 100.0)
oil_viscosity = st.sidebar.number_input("Oil Viscosity", 0.0, 100.0, 10.0)
humidity = st.sidebar.number_input("Ambient Humidity (%)", 0.0, 100.0, 40.0)

operator_exp = st.sidebar.selectbox(
    "Operator Experience",
    ["Junior", "Mid", "Senior"]
)

last_service_days = st.sidebar.number_input(
    "Days Since Last Service", 0, 1000, 30
)

fault_code = st.sidebar.selectbox(
    "Fault Code",
    ["None", "E101", "E202"]
)

working_hours = st.sidebar.number_input(
    "Total Working Hours", 0, 50000, 1000
)

predict_button = st.sidebar.button("Predict Machine Health")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Explainability", "Model Info"])

if predict_button:

    # --------------------------------------------------
    # Feature Engineering (Must Match Training Logic)
    # --------------------------------------------------
    stress_index = torque * vibration
    rolling_avg_temp = temp  # single input case
    machine_age = working_hours

    exp_map = {"Junior": 1, "Mid": 2, "Senior": 3}
    operator_level = exp_map[operator_exp]

    # --------------------------------------------------
    # Build Input DataFrame (MATCH TRAINING FEATURES)
    # --------------------------------------------------
    input_df = pd.DataFrame({
        "Machine_Model": [machine_model],
        "Avg_Temperature": [temp],
        "Vibration_Level": [vibration],
        "Rotating_Speed": [rot_speed],
        "Voltage_Fluctuation": [voltage],
        "Torque_Nm": [torque],
        "Oil_Viscosity": [oil_viscosity],
        "Ambient_Humidity": [humidity],
        "Operator_Experience": [operator_exp],
        "Last_Service_Days": [last_service_days],
        "Fault_Code": [fault_code],
        "Working_Hours_Total": [working_hours],
        "Rolling_Avg_Temp": [rolling_avg_temp],
        "Stress_Index": [stress_index],
        "Machine_Age": [machine_age],
        "Operator_Experience_Level": [operator_level]
    })

    # --------------------------------------------------
    # Preprocess Input
    # --------------------------------------------------
    X_input = preprocessor.transform(input_df)

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------
    probability = model.predict_proba(X_input)[0][1]
    predicted_class = model.predict(X_input)[0]
    rul_prediction = rul_model.predict(X_input)[0]

    # --------------------------------------------------
    # TAB 1 - OVERVIEW
    # --------------------------------------------------
    with tab1:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Failure Probability")
            st.metric("Probability of Failure (Next 24H)", f"{probability:.2%}")
            st.progress(int(probability * 100))

        with col2:
            st.subheader("Remaining Useful Life")
            st.metric("Estimated Remaining Hours", f"{rul_prediction:.0f} hrs")

        st.subheader("Machine Health Status")

        if predicted_class == 0:
            st.success("Healthy - No immediate maintenance required.")
        else:
            st.error("High Risk of Failure - Immediate maintenance required.")

        if rul_prediction > 1000:
            st.info("Machine has sufficient remaining life.")
        elif rul_prediction > 300:
            st.warning("Preventive maintenance recommended soon.")
        else:
            st.error("Remaining useful life critically low.")

    # --------------------------------------------------
    # TAB 2 - SHAP
    # --------------------------------------------------
    with tab2:

        st.subheader("Prediction Explanation (SHAP)")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)

            fig, ax = plt.subplots()

            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_input[0],
                    feature_names=preprocessor.get_feature_names_out()
                ),
                show=False
            )

            st.pyplot(fig)

        except Exception:
            st.write("SHAP explanation not available for this model type.")

    # --------------------------------------------------
    # TAB 3 - MODEL INFO
    # --------------------------------------------------
    with tab3:

        st.subheader("System Summary")

        st.write("• Classification model predicts probability of machine failure within the next 24 hours.")
        st.write("• Regression model estimates Remaining Useful Life (RUL).")
        st.write("• Feature engineering includes Stress Index and rolling metrics.")
        st.write("• Health status is determined using the model's predicted class output.")
        st.write("• SHAP is used for local explainability of predictions.")

else:
    st.info("Enter machine parameters and click 'Predict Machine Health'.")

st.markdown("---")
st.write("Industrial Predictive Maintenance powered by Machine Learning.")
