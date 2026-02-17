import pytest
import sys
import os
from unittest.mock import MagicMock

# 1. MOCK DAGSHUB & MLFLOW BEFORE IMPORTING TRAIN.PY
# This prevents the "Authorization Required" crash on import
sys.modules["dagshub"] = MagicMock()
sys.modules["mlflow"] = MagicMock()

# 2. Add the project root to path so we can import 'prediction' and 'training'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

@pytest.fixture
def mock_model_files(tmp_path):
    """Creates fake model artifacts for prediction tests"""
    # We fake the artifact directory
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    
    # Create fake model/scaler files
    model_path = artifacts_dir / "model.pkl"
    scaler_path = artifacts_dir / "scaler.pkl"
    model_path.touch()
    scaler_path.touch()
    
    return str(artifacts_dir)