from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, mock_open
import sys

# 1. Prevent "Authorization Required" crash
sys.modules["dagshub"] = MagicMock()
sys.modules["mlflow"] = MagicMock()

from training.train import app

client = TestClient(app)

@patch("training.train.load_iris")
@patch("training.train.train_test_split")
@patch("training.train.GridSearchCV")
@patch("training.train.joblib.dump") 
@patch("training.train.os.makedirs")
@patch("builtins.open", new_callable=mock_open) # <--- NEW: Mock the file writing!
def test_training_endpoint(mock_file, mock_dirs, mock_dump, mock_grid, mock_split, mock_data):
    """Test the full training pipeline with mocks"""
    
    # 1. Setup Fake Data
    mock_data_obj = MagicMock()
    # We need enough data to avoid scaler errors, but logic is mocked anyway
    mock_data_obj.data = [[1, 2, 3, 4]] * 10 
    mock_data_obj.target = [0, 1] * 5
    mock_data.return_value = mock_data_obj
    
    # Mock split return values (X_train, X_test, y_train, y_test)
    # Note: We provide list-of-lists for X to satisfy StandardScaler
    mock_split.return_value = (
        [[1,2]], [[1,2]], [0], [0] 
    )

    # 2. Setup Fake Grid Search
    mock_grid_instance = mock_grid.return_value
    mock_grid_instance.best_estimator_.predict.return_value = [0] 
    mock_grid_instance.best_params_ = {"C": 1.0}

    # 3. Call the API
    response = client.post("/train")

    # 4. Validation
    # Use assertions to debug if it fails again
    assert response.status_code == 200, f"Failed with: {response.text}"
    
    json_response = response.json()
    assert json_response["status"] == "success"
    assert "accuracy" in json_response