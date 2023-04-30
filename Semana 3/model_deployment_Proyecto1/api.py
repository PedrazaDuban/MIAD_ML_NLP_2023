#!/usr/bin/python

from flask import Flask, request, jsonify
from flask_restplus import Api, Resource, fields
import joblib



app = Flask(__name__)

#Cargamos el Modelo Entrenado
MODEL = joblib.load('VehiclePricePrediction.pkl')

# Selección de características relevantes

MODEL_features= ['Year','Mileage','State','Make','Model']

@app.route('/predict')
def predict():
    Year = request.args.get('Year')
    Mileage = request.args.get('Mileage')
    State = request.args.get('State')
    Make = request.args.get('Make')
    Model = request.args.get('Model')

    # La lista de caracteristicas que se utilizaran
    # para la predicción
    features = [['Year','Mileage','State','Make','Model']]
    
    # Utilizamos el modelo para la predicción de los datos
    label_index = MODEL.predict(features)
   
    label = MODEL_features[label_index[0]]
    
    # Creamos y enviamos la respuesta al cliente
    return jsonify(status='Completed Prediction', prediccion=label)

if __name__ == '__main__':
    # Iniciamos el servidor
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
    
