from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.ml_project.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")



@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 




@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])
            Id =int(request.form['Id'])
            
            # Print inputs to debug
            print(f'Inputs: {fixed_acidity}, {volatile_acidity}, {citric_acid}, {residual_sugar}, {chlorides}, {free_sulfur_dioxide}, {total_sulfur_dioxide}, {density}, {pH}, {sulphates}, {alcohol},{Id}')

            # Prepare the data for prediction
            data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol,Id]
            data = np.array(data).reshape(1, 11)
            
            # Create a PredictionPipeline object and make a prediction
            obj = PredictionPipeline()
            predict = obj.predict(data)

            # Print prediction to debug
            print(f'Prediction: {predict}')

            return render_template('results.html', prediction=str(predict))

        except ValueError as ve:
            print('ValueError:', ve)
            return 'Invalid input value. Please check your inputs.'
        except Exception as e:
            print('Exception message:', e)
            return 'Something went wrong. Please try again.'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)