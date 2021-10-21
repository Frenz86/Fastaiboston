import joblib
from pydantic import BaseModel
from typing import Optional

model = joblib.load('BostonClassifier.pkl')

class Feature_type(BaseModel):
    input_1: float =2.1
    input_2: float =2.1
    input_3: float =2.1
    input_4: float =2.1
    input_5: float =2.1
    input_6: float =2.1
    input_7: float =2.1
    input_8: float =2.1
    input_9: float =2.1
    input_10: float =2.1
    input_11: float =2.1
    input_12: float =2.1
    input_13: float =2.1 
    #startDatetime:  Optional[float] =  None

    # CRIM: float
    # ZN: float
    # INDUS: float
    # CHAS: float
    # NOX: float
    # RM: float
    # AGE: float
    # DIS: float
    # RAD: float
    # TAX: float
    # PTRATIO: float
    # B: float
    # LSTAT: float