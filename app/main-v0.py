from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('xgb_model_trained.pkl')  # Ensure the model is loaded correctly

class PredictionInput(BaseModel):
    features: list  # List of lists, where each sublist is a set of features
    class Config:
        schema_extra = {
            "example": {
                "features": [[25, "admin.", "married", "high.school", "no", "yes", "no", "telephone", "jun", "mon", 2, 999, 1, "failure", 93.994, -36.4, 5191]]
            }
        }



@app.post("/predict/")
async def make_prediction(input: PredictionInput):
    try:
        # Create a DataFrame from the input features
        # The column names must match exactly with those used during the model training
        columns = columns = ["age", "job", "marital", "education", "default", "housing", "loan",
                            "contact", "month", "day_of_week", "campaign", "pdays", "previous",
                            "poutcome", "cons.price.idx", "cons.conf.idx", "nr.employed"]
        data = pd.DataFrame(input.features, columns=columns)
        
        # Ensure the DataFrame shape matches the expected input shape for your model
        if data.shape[1] != len(columns):
            return {"error": f"Expected {len(columns)} features, but got {data.shape[1]}"}

        # Predict using the loaded model
        prediction = model.predict(data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}