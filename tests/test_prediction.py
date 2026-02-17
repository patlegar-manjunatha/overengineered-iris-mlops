from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from prediction.main import app 

client = TestClient(app)

# --- TEST 1: HOME PAGE ---
def test_home_endpoint():
    """Ensure the UI loads successfully (Status 200)"""
    response = client.get("/")
    assert response.status_code == 200
    # Check if Jinja2 rendered the template (look for key text)
    assert "NEXUS" in response.text

# --- TEST 2: PREDICTION LOGIC ---
@patch("prediction.main.joblib.load") # <--- Fake the model loader
@patch("prediction.main.os.path.exists")
def test_predict_endpoint(mock_exists, mock_joblib):
    """Test if /predict accepts data and returns a result"""
    
    # 1. Setup the fake model
    mock_exists.return_value = True # Pretend files exist
    
    # Create a fake model object that returns class '0' (Setosa)
    fake_model = MagicMock()
    fake_model.predict.return_value = [0] 
    
    # Create a fake scaler
    fake_scaler = MagicMock()
    fake_scaler.transform.return_value = [[1.0, 2.0, 3.0, 4.0]]

    # Tell joblib to return our fakes
    mock_joblib.side_effect = [fake_model, fake_scaler]

    # 2. Make the Request
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    # Note: We use data=payload because it's a Form, not JSON
    response = client.post("/predict", data=payload)

    # 3. Validation
    assert response.status_code == 200
    assert "Setosa" in response.text

# --- TEST 3: TRIGGER TRAINING (MOCKING HTTPX) ---
@patch("prediction.main.httpx.AsyncClient") 
def test_trigger_training_success(mock_client):
    """Test if the button triggers the training service"""
    
    # 1. Mock the response from the 'Training' container
    mock_response = MagicMock()
    mock_response.status_code = 200
    
    # Setup the async context manager mess
    mock_client_instance = mock_client.return_value.__aenter__.return_value
    mock_client_instance.post.return_value = mock_response

    # 2. Call the endpoint
    with patch("prediction.main.load_artifacts", return_value=True):
        response = client.post("/trigger-train")

    # 3. Validation
    assert response.status_code == 200
    assert response.json()["message"] == "Training successful & Model reloaded"