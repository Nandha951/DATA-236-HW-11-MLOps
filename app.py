from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Define the path to the saved model
MODEL_PATH = os.path.join("models", "model.pkl")

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please run train_model.py first.")
    model = None # Set model to None if loading fails
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the input data model based on Iris dataset features
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "MLOps FastAPI service is running. Go to /docs for API documentation."}

@app.post("/predict")
def predict_iris(features: IrisFeatures):
    """
    Predicts the Iris species based on input features.
    """
    if model is None:
        return {"error": "Model not loaded. Cannot make predictions."}

    # Convert input features to a numpy array
    data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])

    # Make prediction
    prediction = model.predict(data).tolist() # Convert numpy array to list for JSON response

    # Assuming Iris dataset target names for interpretation
    # You might need to adjust this based on your specific model/dataset
    iris_target_names = ["setosa", "versicolor", "virginica"]
    predicted_species = iris_target_names[prediction[0]] if prediction and 0 <= prediction[0] < len(iris_target_names) else "unknown"


    return {"prediction": prediction[0], "predicted_species": predicted_species}

if __name__ == "__main__":
    import uvicorn
    # To run the app, use: uvicorn app:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)