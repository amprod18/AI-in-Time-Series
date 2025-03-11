# tests/test_data.py
import pytest
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from AI_in_Time_Series.data.SyntheticDatasets import _apply_scalers
from AI_in_Time_Series import DatasetStorer
from AI_in_Time_Series.data import create_synthetic_dataset, generate_synthetic_dataset

# Mock SampleODEs class for testing
class MockSampleODEs:
    def __init__(self, **kwargs):
        self.params = kwargs

    def exponential_decay(self, t, x):
        return -0.5 * x  # Simple decay function for testing

# Fixture for mock ODE parameters
@pytest.fixture
def default_ode_params():
    return {
        "ode_func": MockSampleODEs().exponential_decay,
        "start_t": 0.0,
        "max_t": 10.0,
        "n_points": 100,
        "look_ahead": 1,
        "initial_guess": torch.tensor([1.0, 1.0]),
        "method": "rk4",
        "ode_params": {}
    }

# Fixture for sample DataFrame
@pytest.fixture
def sample_df():
    X = pd.DataFrame({"t": range(5), "Component1": [1, 2, 3, 4, 5]})
    y = pd.DataFrame({"Component1_target": [2, 4, 6, 8, 10]})
    return X, y

# --- Tests for _apply_scalers ---
def test_apply_scalers_no_scalers(sample_df):
    X, y = sample_df
    X_scaled, y_scaled, sc_x, sc_y = _apply_scalers(X, y, scalers=None)
    assert X_scaled.shape == X.shape
    assert y_scaled.shape == y.shape
    assert isinstance(sc_x, MinMaxScaler)
    assert isinstance(sc_y, MinMaxScaler)
    assert (X_scaled <= 1).all().all()  # MinMaxScaler bounds

def test_apply_scalers_with_scalers(sample_df):
    X, y = sample_df
    sc_x, sc_y = MinMaxScaler().fit(X), MinMaxScaler().fit(y)
    X_scaled, y_scaled, sc_x_out, sc_y_out = _apply_scalers(X, y, scalers=(sc_x, sc_y))
    assert X_scaled.shape == X.shape
    assert sc_x_out is sc_x  # Should return the same scaler objects

# Edge Case: Empty DataFrame
def test_apply_scalers_empty_df():
    X = pd.DataFrame(columns=["t", "Component1"])
    y = pd.DataFrame(columns=["Component1_target"])
    with pytest.raises(ValueError):
        _apply_scalers(X, y, scalers=None)

# Edge Case: Mismatched columns with pre-fitted scalers
def test_apply_scalers_mismatched_columns(sample_df):
    X, y = sample_df
    sc_x = MinMaxScaler().fit(pd.DataFrame({"wrong_col": [1, 2, 3]}))
    with pytest.raises(ValueError):  # sklearn should raise this
        _apply_scalers(X, y, scalers=(sc_x, MinMaxScaler()))

# --- Tests for generate_synthetic_dataset ---
def test_generate_synthetic_dataset_default(default_ode_params):
    result = generate_synthetic_dataset(**default_ode_params)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (100, 3)  # t + 2 components
    assert torch.all(result[:, 0] == torch.arange(0, 10, 0.1))  # t values

def test_generate_synthetic_dataset_euler(default_ode_params):
    default_ode_params["method"] = "euler"
    result = generate_synthetic_dataset(**default_ode_params)
    assert result.shape == (100, 3)

# Edge Case: n_points = 1
def test_generate_synthetic_dataset_single_point(default_ode_params):
    default_ode_params["n_points"] = 1
    result = generate_synthetic_dataset(**default_ode_params)
    assert result.shape == (1, 3)  # Still includes t and initial guess

# Edge Case: Invalid method
def test_generate_synthetic_dataset_invalid_method(default_ode_params):
    default_ode_params["method"] = "invalid"
    with pytest.raises(ValueError):  # torchdiffeq should raise this
        generate_synthetic_dataset(**default_ode_params)

# Edge Case: Empty initial_guess
def test_generate_synthetic_dataset_empty_initial_guess(default_ode_params):
    default_ode_params["initial_guess"] = torch.tensor([])
    with pytest.raises(ValueError):  # Suggest adding validation
        generate_synthetic_dataset(**default_ode_params)

# --- Tests for create_synthetic_dataset ---
def test_create_synthetic_dataset_default(default_ode_params):
    dataset = create_synthetic_dataset(default_ode_params)
    assert isinstance(dataset, DatasetStorer)
    assert hasattr(dataset, "data")
    assert dataset.data.X_train_scaled.shape == (80, 3)  # 80% of 100
    assert dataset.data.X_test_scaled.shape == (20, 3)
    assert hasattr(dataset, "scalers")

# Edge Case: Zero look_ahead
def test_create_synthetic_dataset_zero_lookahead(default_ode_params):
    default_ode_params["look_ahead"] = 0
    dataset = create_synthetic_dataset(default_ode_params)
    assert dataset.data.y_train_scaled.shape[1] == 2  # Still works, predicts same step

# Edge Case: n_points too small for split
def test_create_synthetic_dataset_small_n_points(default_ode_params):
    default_ode_params["n_points"] = 2
    dataset = create_synthetic_dataset(default_ode_params)
    assert dataset.data.X_train_scaled.shape[0] >= 1  # At least 1 train point
    assert dataset.data.X_test_scaled.shape[0] >= 0  # Could be empty

# Edge Case: Invalid ode_func
def test_create_synthetic_dataset_invalid_ode_func(default_ode_params):
    default_ode_params["ode_func"] = "not_a_function"
    with pytest.raises(AttributeError):  # Due to getattr in generate_synthetic_dataset
        create_synthetic_dataset(default_ode_params)

# Edge Case: Empty ode_params
def test_create_synthetic_dataset_empty_ode_params(default_ode_params):
    default_ode_params["ode_params"] = None
    dataset = create_synthetic_dataset(default_ode_params)
    assert isinstance(dataset, DatasetStorer)  # Should still work with defaults