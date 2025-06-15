import numpy as np 
import joblib
import streamlit 

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
    streamlit.title("Diabetes Prediction Web App")


    Pregnancies=streamlit.text_input("Number of Pregnancies")
    Glucose=streamlit.text_input("Glucose Level")
    BloodPressure=streamlit.text_input("Blood Pressure value")
    SkinThickness=streamlit.text_input("Skin Thickness value")
    Insulin=streamlit.text_input("Insulin Level")
    BMI=streamlit.text_input("BMI value")
    DiabetesPedigreeFunction=streamlit.text_input("Diabetes Pedigree Function value")
    Age=streamlit.text_input("Age of the Person")

    diagnosis=""

    if streamlit.button("Diabetes Test Result"):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    

    streamlit.success(diagnosis)




if __name__=="__main__":
        main()
