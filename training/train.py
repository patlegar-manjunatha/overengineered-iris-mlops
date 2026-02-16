import matplotlib
matplotlib.use("Agg")
from fastapi import FastAPI, HTTPException
import uvicorn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib
import mlflow
import dagshub

app = FastAPI(title="Iris Training Service")

dagshub.init(repo_owner='patlegar-manjunatha', repo_name='overengineered-iris-mlops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/patlegar-manjunatha/overengineered-iris-mlops.mlflow/")
mlflow.set_experiment('LogisticRegression')


def load_data():
    try : 
        data = load_iris()
        print('Data loaded succesfully')
        return data
    except Exception as e : 
        print('Error occured during load_data : ', e) 
        raise e

def get_grid():
    return [
        {
            "solver": ["saga"],
            "penalty": ["elasticnet"],
            "C": [0.01, 1, 100],
            "l1_ratio": [0, 0.5, 1],
            "max_iter": [5000],
        }
    ]


def train_model():

    data = load_data() 
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    param_grid = get_grid()
    model = LogisticRegression()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, n_jobs=-1)
    
    mlflow.sklearn.autolog(log_models=True, log_input_examples=True, exclusive=False)

    with mlflow.start_run(run_name='GridSearch_best_estimator') as run: 
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", test_accuracy)

        report = classification_report(y_test, y_pred)
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/classification_report.txt', 'w') as f: 
            f.write(report)
        mlflow.log_artifact('metrics/classification_report.txt')


        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(best_model, 'artifacts/model.pkl')
        joblib.dump(scaler, 'artifacts/scaler.pkl')

    print("Training process is completed")
    return {'status' : 'success', 'accuracy' : test_accuracy, 'best_params' : grid_search.best_params_}

@app.post("/train")
def trigger_training():
    try: 
        result = train_model()
        return result
    except Exception as e : 
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == '__main__': 
    uvicorn.run(app, host="0.0.0.0", port=8001)