from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_and_save_model():
    """Trains a Logistic Regression model on the Iris dataset and saves it."""
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    print("Dataset loaded.")

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    print("Model training complete.")

    # Create a directory for the model if it doesn't exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")

    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    train_and_save_model()