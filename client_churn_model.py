# Databricks notebook source
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, rocmodel_path_auc_score, confusion_matrix, classification_report, roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ClientChurnModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = [
            'course', 'duration', 'method_of_delivery', 'subscription_type',
            'location', 'num_users', 'price'
        ]

    def generate_sample_data(self, n_samples=1000):
        """Generate realistic client data for training"""
        np.random.seed(42)

        # Define possible values for categorical features
        courses = ['Leadership Development', 'Management Skills', 'Communication', 'Project Management', 'Data Analysis']
        delivery_methods = ['Online', 'In-person', 'Hybrid']
        subscription_types = ['Monthly', 'Annual', 'One-time']
        locations = ['North America', 'Europe', 'Asia', 'South America', 'Africa']

        data = {
            'course': np.random.choice(courses, n_samples),
            'duration': np.random.randint(1, 53, n_samples),  # Weeks
            'method_of_delivery': np.random.choice(delivery_methods, n_samples),
            'subscription_type': np.random.choice(subscription_types, n_samples),
            'location': np.random.choice(locations, n_samples),
            'num_users': np.random.randint(1, 100, n_samples),  # Number of users
            'price': np.random.uniform(100, 5000, n_samples)  # Price in dollars
        }

        # Generate churn label (1 = churn, 0 = not churn)
        # For demonstration, we'll create a simple relationship
        churn_prob = (
            0.1 + 0.02 * data['duration'] +
            0.1 * (data['price'] > 2000) +
            0.1 * (data['num_users'] > 50) +
            0.1 * (data['method_of_delivery'] == 'Online')
        )
        churn_prob = np.clip(churn_prob, 0, 1)  # Ensure probabilities are between 0 and 1
        churn = np.random.binomial(1, churn_prob)

        df = pd.DataFrame(data)
        df['churn'] = churn

        return df

    def load_data(self, file_path=None):
        """Load data from file or generate sample data"""
        if file_path and pd.io.common.file_exists(file_path):
            self.data = pd.read_csv(file_path)
        else:
            print("Generating sample client data...")
            self.data = self.generate_sample_data()
            self.data.to_csv('client_data.csv', index=False)
            print("Sample data saved to client_data.csv")

        return self.data

    def preprocess_data(self):
        """Preprocess the data for training"""
        # Handle missing values
        self.data = self.data.fillna(0)

        # Define preprocessing for numeric and categorical features
        numeric_features = ['duration', 'num_users', 'price']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_features = ['course', 'method_of_delivery', 'subscription_type', 'location']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Select features for training
        X = self.data[self.feature_names]
        y = self.data['churn']

        return X, y

    def train_model(self, model_type='random_forest'):
        """Train the client churn prediction model"""
        X, y = self.preprocess_data()

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Preprocess the data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )

        self.model.fit(X_train_processed, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test_processed)
        y_pred_proba = self.model.predict_proba(X_test_processed)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Store test data for evaluation
        self.X_test = X_test_processed
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }

    def predict_churn(self, client_features):
        """Predict churn for given client features"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        # Convert to DataFrame if it's a dictionary
        if isinstance(client_features, dict):
            # Create DataFrame with all required features
            df = pd.DataFrame([client_features])

            # Ensure all features are present
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0

            X = df[self.feature_names]
        else:
            X = client_features

        # Preprocess the data
        X_processed = self.preprocessor.transform(X)

        # Make prediction
        prediction = self.model.predict(X_processed)
        prediction_proba = self.model.predict_proba(X_processed)[:, 1]

        return {
            'churn': bool(prediction[0]),
            'churn_probability': prediction_proba[0]
        }

    def get_feature_importance(self):
        """Get feature importance for Random Forest model"""
        if hasattr(self.model, 'feature_importances_'):
            # Get feature names from the preprocessor
            feature_names = self.preprocessor.get_feature_names_out()

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None

    def save_model(self, model_path='churn_model.pkl', preprocessor_path='preprocessor.pkl'):
        """Save the trained model and preprocessor"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        joblib.dump(self.model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Model saved to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")

    def load_model(self, model_path='churn_model.pkl', preprocessor_path='preprocessor.pkl'):
        """Load a pre-trained model and preprocessor"""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        print("Model and preprocessor loaded successfully")

    def plot_confusion_matrix(self):
        """Plot confusion matrix for the model"""
        if not hasattr(self, 'y_test'):
            print("No test data available. Train the model first.")
            return

        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Churn', 'Churn'],
                    yticklabels=['Not Churn', 'Churn'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(self):
        """Plot ROC curve for the model"""
        if not hasattr(self, 'y_test') or not hasattr(self, 'y_pred_proba'):
            print("No test data available. Train the model first.")
            return

        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, marker='.', label='Random Forest')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Example usage
    model = ClientChurnModel()

    # Load or generate data
    data = model.load_data()
    print(f"Data shape: {data.shape}")
    print("\nFirst few rows:")
    print(data.head())

    # Train the model
    metrics = model.train_model()

    # Save the model
    model.save_model()

    # Example prediction
    sample_client = {
        'course': 'Leadership Development',
        'duration': 12,  # Weeks
        'method_of_delivery': 'Online',
        'subscription_type': 'Monthly',
        'location': 'North America',
        'num_users': 25,
        'price': 1500  # Price in dollars
    }

    churn_prediction = model.predict_churn(sample_client)
    print(f"\nPredicted churn for sample client: {churn_prediction['churn']}")
    print(f"Churn probability: {churn_prediction['churn_probability']:.4f}")

    # Show feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10))

    # Plot evaluation metrics
    model.plot_confusion_matrix()
    model.plot_roc_curve()


# COMMAND ----------

import os

model_path = 'churn_model.pkl'
model_full_path = os.path.abspath(model_path)
model_full_path

# COMMAND ----------

