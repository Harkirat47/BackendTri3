from flask import Blueprint, request, jsonify
from model.train import titanic_model

predict_api = Blueprint('predict_api', __name__)

@predict_api.route('/predict', methods=['POST'])
def predict():
    # Get passenger data from request
    passenger_data = request.json

    # Preprocess passenger data
    # Apply the same preprocessing steps used during training

    # Make predictions using trained models
    logreg, dt, encoder, X_train, X_test, y_train, y_test = titanic_model.get_models()
    # Perform predictions using logreg or dt based on your requirement

    # Return prediction results
    return jsonify({'prediction': prediction_result})