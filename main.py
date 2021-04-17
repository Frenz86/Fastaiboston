from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List

import numpy as np
import pandas as pd
import json
import joblib

# Variabili globali
loaded_model = None

# CORS Support
origins = [
    f'http://0.0.0.0:8008',
    f'http://localhost:8008',
    #f'http://localhost:8080',
]

app = FastAPI()

app.add_middleware(CORSMiddleware,
                    allow_origins=origins,
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                    )

# Template setup
templates = Jinja2Templates(directory="templates")

@app.get("/")
@app.get("/test", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request" : request})

@app.on_event("startup")
async def startup_event():
    print("Start")
    # caricamento modelli
    # load all stuff
    global loaded_model
    loaded_model = joblib.load('BostonClassifier.pkl')

async def shutdon_envent():
    print("Shutdown")

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

@app.post('/predict',response_class=JSONResponse )
def predict_species(input_: InputModel):
    data = input_.dict()
    data_in = [[data['input_1'], data['input_2'], data['input_3'], data['input_4'],
                data['input_5'], data['input_6'], data['input_7'], data['input_8'],
                data['input_9'], data['input_10'], data['input_11'], data['input_12'],data['input_13']]]
    
    prediction = loaded_model.predict(data_in)
    #probability = loaded_model.predict_proba(data_in).max()
    return json.dumps({
                        'prediction': prediction[0],
                        #'probability': probability
                        })