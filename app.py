import numpy as np
import pickle
import streamlit as st
import sklearn 
from sklearn.preprocessing import StandardScaler

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

scalar = StandardScaler()

def Diabetes_prediction(input_data):

    inp_data = np.asarray(input_data)

    inp_datas = inp_data.reshape(1,-1)

    prediction = loaded_model.predict(scalar.fit_transform(inp_datas))

    if (prediction==0):
        return 'Not Diabetic'
    else:
        return 'Diabetic'
    
def main():
    st.title('Diabetic Prediction')
    left,right = st.columns([1,1])
    with left:
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Glucose Level')
        BloodPressure = st.text_input('Blood Pressure Value')
        SkinThickness = st.text_input('Skin Thickness Value')
    with right:
        Insulin = st.text_input('Insulin Level')
        BMI =  st.text_input('BMI Value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        Age = st.text_input('Age of the Person')

    diagnosis = ''

    if st.button('Diabetes Test Result'):
        diagnosis = Diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()
    
