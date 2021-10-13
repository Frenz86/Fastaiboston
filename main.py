from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

import numpy as np
import pandas as pd
import json

from model import InputModel, loaded_model


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

# @app.on_event("startup")
# async def startup_event():
#     print("Start")
#     # caricamento modelli
#     # load all stuff
#     global loaded_model

async def shutdon_envent():
    print("Shutdown")



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

if __name__ == '__main__':
	uvicorn.run(app,host="127.0.0.1",port=8000)