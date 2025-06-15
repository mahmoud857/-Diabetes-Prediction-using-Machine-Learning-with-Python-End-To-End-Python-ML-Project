import numpy as np
import joblib
import streamlit as st

# تحميل النموذج
loading_model = joblib.load(open("diabetes_model.joblib", "rb"))


def diabetes_prediction(input_data):
    # تحويل الإدخال إلى صفيف NumPy وإعادة تشكيله
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # التنبؤ
    prediction = loading_model.predict(input_data_reshaped)

    # العودة بالنتيجة
    if prediction[0] == 0:
        return "✅ The person is not diabetic"
    else:
        return "⚠️ The person is diabetic"


def main():
    st.title("🩺 Diabetes Prediction Web App")
    st.markdown("### Use the sliders below to enter your health details:")

    # استخدام Slider لكل حقل
    Pregnancies = st.slider("🤰 Number of Pregnancies", min_value=0, max_value=20, value=0,
                            help="Number of times pregnant")
    Glucose = st.slider("🍬 Glucose Level", min_value=0, max_value=200, value=120,
                        help="Plasma glucose concentration after 2 hours in oral test")
    BloodPressure = st.slider("💉 Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70,
                              help="Diastolic blood pressure")
    SkinThickness = st.slider("📏 Skin Thickness (mm)", min_value=0, max_value=100, value=20,
                              help="Triceps skin fold thickness")
    Insulin = st.slider("🧬 Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80,
                        help="2-Hour serum insulin")
    BMI = st.slider("⚖️ BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                     help="Body mass index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction = st.slider("🧬 Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01,
                                        help="Likelihood of diabetes based on family history")
    Age = st.slider("🎂 Age (years)", min_value=0, max_value=120, value=30,
                    help="Age in years")

    diagnosis = ""

    if st.button("🧪 Check Diagnosis"):
        input_data = [
            float(Pregnancies),
            float(Glucose),
            float(BloodPressure),
            float(SkinThickness),
            float(Insulin),
            float(BMI),
            float(DiabetesPedigreeFunction),
            float(Age)
        ]

        diagnosis = diabetes_prediction(input_data)

        # عرض النتيجة بشكل جذاب
        if "not" in diagnosis:
            st.success(diagnosis)
        else:
            st.error(diagnosis)

        # إنشاء تقرير بسيط
        report = f"""
        ==== Diabetes Prediction Report ====
        
        Pregnancies: {input_data[0]}
        Glucose Level: {input_data[1]}
        Blood Pressure: {input_data[2]}
        Skin Thickness: {input_data[3]}
        Insulin Level: {input_data[4]}
        BMI: {input_data[5]:.2f}
        Diabetes Pedigree Function: {input_data[6]:.2f}
        Age: {input_data[7]}

        Diagnosis: {diagnosis}
        
        ====================================
        """

        # زر تنزيل التقرير
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="diabetes_prediction_report.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()
