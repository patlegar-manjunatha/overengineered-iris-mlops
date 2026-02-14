from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Form
from pydantic import BaseModel
import joblib

class InputSchema(BaseModel): 
    sepal_length : float 
    sepal_width : float
    petal_length : float
    petal_width : float


app = FastAPI()

templates = Jinja2Templates (directory="templates")
app.mount('/static', StaticFiles(directory='static'), name='static')

@app.get('/')
def home(request: Request): 
    return templates.TemplateResponse('home.html', {"request": request})

@app.get('/predict-form')
def start(request: Request): 
    return templates.TemplateResponse('predict_form.html', {"request" : request})

@app.post('/predict')
def predict(request : Request, 
            sepal_length : float = Form(...), 
            sepal_width : float = Form(...), 
            petal_length : float = Form(...), 
            petal_width : float = Form(...)):
    input_data = InputSchema(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width)
    features = [[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]]
    target_names = ['setosa', 'versicolor', 'virginica']
    model = joblib.load('artifacts/model.pkl')
    scaler = joblib.load('artifacts/scaler.pkl')
    prediction = model.predict(scaler.transform(features))[0]
    prediction_label = target_names[prediction].capitalize()
    return templates.TemplateResponse('result.html', {"request": request, "prediction" : prediction_label})