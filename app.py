from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar el modelo y las características esperadas
model = joblib.load('models/enfermedad_cardiaca.joblib')
expected_features = joblib.load('models/expected_features.pkl')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Obtener datos del formulario
        data = request.form.to_dict()
        data = {key: [value] for key, value in data.items()}
        input_df = pd.DataFrame.from_dict(data)

        # Realizar one-hot encoding con las mismas columnas que en el entrenamiento
        input_df = pd.get_dummies(input_df, columns=['sex', 'cp'])

        # Asegurar que todas las características esperadas estén presentes
        for feature in expected_features:
            if feature not in input_df.columns:
                input_df[feature] = 0

        # Reordenar las columnas según las características esperadas
        input_df = input_df[expected_features]

        # Convertir las columnas a tipo float
        input_df = input_df.astype(float)

        # Hacer la predicción
        prediction = model.predict(input_df)
        result = 'Enfermedad cardíaca' if prediction[0] == 1 else 'No hay enfermedad cardíaca'

        return render_template('result.html', result=result)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)