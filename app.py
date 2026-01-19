import gradio as gr 
import pandas as pd 
import numpy as np 
import pickle 

#loaded model pickle file 
with open("Diabetes_Prediction_Pipeline.pkl", "rb") as file:
    model = pickle.load(file)

#main logic 
def predict_diabetes(
    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
    BMI, DiabetesPedigreeFunction, Age
):
    input_data = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })

    prediction = model.predict(input_data)
    prediction = np.clip(prediction, 0, 1)

    return f"Predicted Diabetes Outcome: {prediction[0]:.2f}"

inputs=[ 
    gr.Slider(0, 20, step=1, label="Number of Pregnancies", value=0),
    gr.Slider(0, 200, step=1, label="Glucose Level", value=120),
    gr.Slider(40, 200, step=1, label="Blood Pressure ( mm Hg )", value=80),
    gr.Slider(0, 99, step=1, label="Skin Thickness (mm)", value=20),
    gr.Slider(0,25, step=1, label="Insulin Level (mu U/ml)", value=20),
    gr.Number( label="BMI (Body Mass Index)"),
    gr.Slider(0.0, 2.42, step=0.01, label="Diabetes Pedigree Function", value=0.5),
    gr.Number(label="Age (years)")
   
]

# interface 
app = gr.Interface(
    fn= predict_diabetes,
    inputs= inputs,
    outputs=gr.Textbox(label="Output"),
    title="Diabetes Prediction",
    description="Predict the Diabetes Outcome of a patient based on various features."
)

#lanuch 
app.launch(share=True)
