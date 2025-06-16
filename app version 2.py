import numpy as np 
import joblib
import streamlit 
from PIL import Image 


loading_model=joblib.load(open("diabetes_model.joblib","rb"))


def diabetes_prediction(input_data):

    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


    print("---------------------------------------------------")

    prediction=loading_model.predict(input_data_reshaped)

    print(prediction)
    print("---------------------------------------------------")

    if prediction[0]==0:
        return "the person is not diabetic"
    else:
        return "the person is diabetic"
print("---------------------------------------------------")

   


def main():

    img=Image.open("img.png")
    streamlit.image(img, caption="Welcome To Diabetes Prediction App", use_container_width=True)

    streamlit.title("")
    streamlit.markdown("Welcome to the Diabetes Prediction App — a tool designed to estimate diabetes risk based on key health indicators.")
    streamlit.markdown("Simply fill in the required information and press 'Diabetes Test Result' to receive an instant prediction,after that you can download this report by pressing 'Download Report' to easily download the report")
    #streamlit.sidebar.markdown("----")
    streamlit.sidebar.markdown("🧑‍💻 Develop by : Mahmoud Hassan Mahmoud")
    streamlit.sidebar.markdown(" Data scientist&Analytics||Machine Learning Engineer")
    streamlit.sidebar.markdown("📧 Email: mahmoud.ai_0016@ai.kfs.edu.eg")
    streamlit.sidebar.markdown("📞 Phone: +2 0 12 77 42 10 63")
    streamlit.sidebar.markdown("🔗 LinkedIn: www.linkedin.com/in/mahmoudhassanmahmoud")
    streamlit.sidebar.markdown("📄 GitHub: https://github.com/mahmoud857")
    streamlit.sidebar.markdown("🌐 Portfilo: https://taplink.cc/mahmoudhassanmahmoud")
    streamlit.sidebar.markdown("🔗 Kaggle: https://www.kaggle.com/mahmoudhassanmahmoud")

    


    Pregnancies=streamlit.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
    Glucose=streamlit.number_input("Glucose Level", min_value=0, max_value=200, value=0, step=1)
    BloodPressure=streamlit.number_input("Blood Pressure value", min_value=0, max_value=200, value=0, step=1)
    SkinThickness=streamlit.number_input("Skin Thickness value", min_value=0, max_value=100, value=0, step=1)
    Insulin=streamlit.number_input("Insulin Level", min_value=0, max_value=900, value=0, step=1)
    BMI=streamlit.number_input("BMI value", min_value=0.0, max_value=70.0, value=0.0, step=0.1)
    DiabetesPedigreeFunction=streamlit.number_input("Diabetes Pedigree Function value", min_value=0.0, max_value=2.5, value=0.0, step=0.01)
    Age=streamlit.number_input("Age of the Person", min_value=0, max_value=120, value=0, step=1)

    diagnosis=""

    if streamlit.button("Diabetes Test Result"):
        input_data=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
        diagnosis=diabetes_prediction(input_data)
        streamlit.success(diagnosis)


        # Report Details 
 
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
    


        #Report download button 
        streamlit.download_button(
             label="Download Report",
             data=report,
             file_name="diabetes_prediction_report.txt",
             mime="text/plain"
          )
    

        streamlit.success("Report generated successfully!")
if __name__=="__main__":
        main()
