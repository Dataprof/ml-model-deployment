# Databricks notebook source
# MAGIC %run ./client_churn_model

# COMMAND ----------

from flask import Flask, request, jsonify
import os
import joblib
import pandas as pd
#from  import ClientChurnModel

app = Flask(__name__)
model = ClientChurnModel()

# Load model if it exists
model_path = '/Workspace/Users/durga.nagaraju.ctr@dot.gov/churn_model.pkl'
preprocessor_path = '/Workspace/Users/durga.nagaraju.ctr@dot.gov/preprocessor.pkl'
if os.path.exists(model_path) and os.path.exists(preprocessor_path):
    model.load_model(model_path, preprocessor_path)

@app.route('/train', methods=['POST'])
def train_model():
    """Train the churn prediction model"""
    data_file = request.files.get('data_file')
    model_type = request.form.get('model_type', 'random_forest')

    if data_file:
        data_file.save('uploaded_data.csv')
        data = pd.read_csv('uploaded_data.csv')
        model.data = data

    metrics = model.train_model(model_type)
    model.save_model()

    return jsonify({
        'status': 'success',
        'message': 'Model trained successfully',
        'metrics': metrics
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict client churn"""
    if not model.model:
        return jsonify({
            'status': 'error',
            'message': 'Model not trained. Please train the model first.'
        }), 400

    client_data = request.json
    prediction = model.predict_churn(client_data)

    return jsonify({
        'status': 'success',
        'prediction': prediction
    }), 200

@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    """Get feature importance"""
    if not model.model or not hasattr(model.model, 'feature_importances_'):
        return jsonify({
            'status': 'error',
            'message': 'Model not trained or does not support feature importance.'
        }), 400

    importance = model.get_feature_importance()
    if importance is not None:
        importance_dict = importance.to_dict(orient='records')
        return jsonify({
            'status': 'success',
            'feature_importance': importance_dict
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': 'Could not retrieve feature importance.'
        }), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a pre-trained model"""
    model_file = request.files.get('model_file')
    preprocessor_file = request.files.get('preprocessor_file')

    if model_file and preprocessor_file:
        model_file.save('uploaded_model.pkl')
        preprocessor_file.save('uploaded_preprocessor.pkl')
        model.load_model('uploaded_model.pkl', 'uploaded_preprocessor.pkl')
        return jsonify({
            'status': 'success',
            'message': 'Model loaded successfully'
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': 'Model and preprocessor files are required.'
        }), 400

@app.route('/status', methods=['GET'])
def status():
    """Check if model is loaded"""
    is_loaded = model.model is not None
    return jsonify({
        'status': 'success',
        'model_loaded': is_loaded
    }), 200

if __name__ == '__main__':
    app.run(debug=True)


# COMMAND ----------

