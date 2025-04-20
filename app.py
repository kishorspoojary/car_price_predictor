import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from flask_cors import CORS
import locale

app = Flask(__name__)
CORS(app)  # Enable CORS for Heroku deployment

# Load the trained model
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))

# Load the cleaned car data
car = pd.read_csv("cleaned_car.csv")

# Set locale to Indian format (with fallback below)
try:
    locale.setlocale(locale.LC_ALL, 'en_IN.UTF-8')
except:
    pass  # We’ll use a fallback function if locale isn’t supported


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())
    companies.insert(0, "select company")

    # Create a mapping: company -> list of car models
    company_model_map = car.groupby('company')['name'].unique().apply(list).to_dict()

    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=year,
        fuel_types=fuel_type,
        company_model_map=company_model_map
    )


@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    # Predict using the model
    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                             columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    predicted_price = int(np.round(prediction[0], 0))

    # Try to format in Indian number system using locale
    try:
        price_str = locale.format_string("%d", predicted_price, grouping=True)
    except:
        # Fallback formatting if 'en_IN' is not supported on the system
        def format_indian(number):
            num_str = str(number)[::-1]
            result = num_str[:3]
            num_str = num_str[3:]
            for i in range(0, len(num_str), 2):
                result += ',' + num_str[i:i+2]
            return result[::-1]

        price_str = format_indian(predicted_price)

    return f"{price_str}"


if __name__ == "__main__":
    app.run(debug=True)
