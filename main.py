from fastapi import FastAPI,Request,Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
from model import Feature_type, model
import json


# CORS Support
origins = [f'http://0.0.0.0:8000',
           f'http://localhost:8000',
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
def link():
    return{"-->        http://localhost:8000/docs   http://localhost:8000/html            <-----"}


@app.get("/html", response_class=HTMLResponse)
async def html(request: Request):
    return templates.TemplateResponse("index.html", {"request" : request})

@app.get("/predict")
async def predict_get(data: Feature_type= Depends()):              # input nel corpo
    try:
        data = pd.DataFrame(data)
        data = data.T
        data.rename(columns=data.iloc[0], inplace = True)
        data= data.iloc[1:]
        predictions = model.predict(data)
        return {'prediction': predictions[0]}
    except:
        return {"prediction": "there was an error"} 

@app.post('/predict')
async def predict_post(data: Feature_type):
    try:
        data = pd.DataFrame(data)
        data = data.T
        data.rename(columns=data.iloc[0], inplace = True)
        data= data.iloc[1:]
        predictions = model.predict(data)
        return {"prediction": predictions[0]}
    except:
        return {"prediction": "there was an error"} 

@app.put('/predict')
async def predict_put(data: Feature_type):
    try:
        data = pd.DataFrame(data)
        data = data.T
        data.rename(columns=data.iloc[0], inplace = True)
        data= data.iloc[1:]
        predictions = model.predict(data)
        return {"prediction": predictions[0]}
    except:
        return {"prediction": "there was an error"} 



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)