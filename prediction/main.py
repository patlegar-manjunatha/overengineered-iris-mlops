from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import httpx
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title='Iris Prediction')

templates = Jinja2Templates (directory=os.path.join(BASE_DIR, "templates"))
app.mount('/static', StaticFiles(directory=os.path.join(BASE_DIR, "static")), name='static')

class InputSchema(BaseModel): 
    sepal_length : float 
    sepal_width : float
    petal_length : float
    petal_width : float

model = None
scaler = None
ARTIFACTS_DIR = 'artifacts'
def load_artifacts(): 
    global model, scaler
    try : 
        model_path = os.path.join(ARTIFACTS_DIR, 'model.pkl')
        scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')

        if os.path.exists(model_path) and os.path.exists(scaler_path): 
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return True
        return False
    except Exception as e : 
        print(f"Error loading artifacts : {e}")
        return False

load_artifacts()

@app.get('/')
def home(request: Request): 
    return templates.TemplateResponse(request=request, name='home.html')

@app.get('/predict-form')
def start(request: Request): 
    return templates.TemplateResponse(request=request, name='predict_form.html')

@app.post('/trigger-train')
async def trigger_training(): 
    """
    Calls the Training Container to start the process
    """
    training_service_url = "http://training-service:8001/train"
    try : 
        async with httpx.AsyncClient() as client: 
            response = await client.post(training_service_url, timeout=60.0)
        if response.status_code == 200: 
            if load_artifacts(): 
                return {'message' : "Training successful & Model reloaded"}
            else : 
                return {"message" : "Training finished, but failed to load the new model"}
        else : 
            raise HTTPException(status_code=500, detail='Training Service Failed')
    except Exception as e : 
        raise HTTPException(status_code=500, detail=f'Connection to Training Service failed {e}')

@app.post('/predict')
def predict(request : Request, 
            sepal_length : float = Form(...), 
            sepal_width : float = Form(...), 
            petal_length : float = Form(...), 
            petal_width : float = Form(...)):
    if model is None or scaler is None: 
        if not load_artifacts(): 
            return templates.TemplateResponse('result.html', {'request' : request, 'prediction' : 'Error : Model not found. Please click "Train" first.'})
    input_data = InputSchema(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width)
    features = [[input_data.sepal_length, input_data.sepal_width, input_data.petal_length, input_data.petal_width]]

    scaled_features = scaler.transform(features)
    prediction_idx = model.predict(scaled_features)[0]

    target_names = ['setosa', 'versicolor', 'virginica']
    prediction_label = target_names[prediction_idx].capitalize()
    return templates.TemplateResponse(request=request, name='result.html', context={'prediction' : prediction_label})