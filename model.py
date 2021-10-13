import joblib
from pydantic import BaseModel
from typing import Optional

loaded_model = joblib.load('BostonClassifier.pkl')

class InputModel(BaseModel):
    input_1: float
    input_2: float
    input_3: float
    input_4: float
    input_5: float
    input_6: float
    input_7: float
    input_8: float
    input_9: float
    input_10: float
    input_11: float
    input_12: float
    input_13: float
    startDatetime:  Optional[float] =  None