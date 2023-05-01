import pandas as pd
import joblib
import sys
import os
from sklearn.preprocessing import LabelEncoder

def predict_proba(year, mileage, state, make, model):

    ModeloEntrenado = joblib.load(os.path.dirname(__file__) + '/VehiclePricePrediction.pkl') 

    datos_entrada = pd.DataFrame(
        {
            'Year': [year], 
            'Mileage': [mileage],
            'State': [state],
            'Make': [make],
            'Model': [model]
        }
    )
    print(f'Datos Ingresados desde el API: ')
    print(f'{datos_entrada}')
    print(type(datos_entrada))
    # Create arrary of categorial variables to be encoded
    categorical_cols = ['State', 'Make', 'Model']
    le = LabelEncoder()
    # apply label encoder on categorical feature columns
    datos_entrada[categorical_cols] = datos_entrada[categorical_cols].apply(lambda col: le.fit_transform(col))
    print(f'Datos Ingresados Codificados: ')
    print(f'{datos_entrada}')
    print(type(datos_entrada))
   
    # Hacer predicción
    p1 = ModeloEntrenado.predict(datos_entrada)
    #p1="Entro Aquí" 
    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 5:
        print('Please add the characteristics of the vehicle ')
        
    else:

        year = int(sys.argv[1])
        mileage = int(sys.argv[2])
        state = sys.argv[3]
        make = sys.argv[4]
        model = sys.argv[5]

        p1 = predict_proba(year, mileage, state, make, model)
        
        print(f'Características del vehículo: {year} {make} {model} con {mileage} millas en el {state}')
        print('Probability of Pricing: ', p1)