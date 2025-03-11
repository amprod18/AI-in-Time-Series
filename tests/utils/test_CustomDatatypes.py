# tests/test_models.py
import pytest
import torch
from AI_in_Time_Series import WindowedDataType, DatasetStorer
from AI_in_Time_Series.visualization.plotting import _build_tree_dict
from AI_in_Time_Series.visualization import build_networkx_graph

# Fixture for sample tensor data
@pytest.fixture
def sample_data():
    X = torch.arange(10, dtype=torch.float32)  # [0, 1, 2, ..., 9]
    y = torch.arange(10, dtype=torch.float32) * 2  # [0, 2, 4, ..., 18]
    return X, y

# Fixture for a WindowedDataType instance
@pytest.fixture
def windowed_dataset(sample_data):
    X, y = sample_data
    return WindowedDataType(X, y, window_size=3)

# Fixture for a DatasetStorer instance
@pytest.fixture
def dataset_storer():
    ds = DatasetStorer("root", param1=42)
    ds.add_container("child1", param2="test")
    ds.add_container("child2").child2.add_container("grandchild", param3=3.14)
    return ds


# --- WindowedDataType Tests ---
def test_windowed_dataset_length(windowed_dataset):
    assert len(windowed_dataset) == 7  # 10 - 3 = 7

def test_windowed_dataset_getitem(windowed_dataset):
    X_window, y_target = windowed_dataset[0]
    assert torch.allclose(X_window, torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(y_target, torch.tensor(4.0))  # y[2] = 4

def test_windowed_dataset_out_of_bounds(windowed_dataset):
    with pytest.raises(IndexError):
        windowed_dataset[7]  # len is 7, so index 7 is out of bounds

# Edge Case: Window size equals data length
def test_windowed_dataset_window_equals_length(sample_data):
    X, y = sample_data
    dataset = WindowedDataType(X, y, window_size=10)
    assert len(dataset) == 0  # No valid windows possible

# Edge Case: Window size larger than data length
def test_windowed_dataset_window_larger_than_length(sample_data):
    X, y = sample_data
    with pytest.raises(ValueError): # Suggest adding validation in __init__
        WindowedDataType(X, y, window_size=11)

# Edge Case: Empty tensors
def test_windowed_dataset_empty_tensors():
    X = torch.tensor([])
    y = torch.tensor([])
    with pytest.raises(ValueError):  # Suggest adding validation in __init__
        WindowedDataType(X, y, window_size=1)

# Edge Case: Negative or zero window size
def test_windowed_dataset_invalid_window_size(sample_data):
    X, y = sample_data
    with pytest.raises(ValueError):  # Suggest adding validation in __init__
        WindowedDataType(X, y, window_size=0)
    with pytest.raises(ValueError):
        WindowedDataType(X, y, window_size=-1)

# --- DatasetStorer Tests ---
def test_dataset_storer_init():
    ds = DatasetStorer("test", foo="bar", num=123)
    assert ds.name == "test"
    assert ds.foo == "bar"
    assert ds.num == 123

def test_dataset_storer_add_container(dataset_storer):
    assert hasattr(dataset_storer, "child1")
    assert dataset_storer.child1.name == "child1"
    assert dataset_storer.child1.param2 == "test"
    assert hasattr(dataset_storer.child2, "grandchild")
    assert dataset_storer.child2.grandchild.param3 == 3.14

# Edge Case: Empty DatasetStorer
def test_dataset_storer_empty():
    ds = DatasetStorer("empty")
    assert ds.name == "empty"
    assert len([attr for attr in dir(ds) if not attr.startswith("_") and attr != "name"]) == 1

# Edge Case: Invalid container name (e.g., Python keyword or special chars)
def test_dataset_storer_invalid_container_name():
    ds = DatasetStorer("root")
    # This might not raise an error by default, but it’s a potential issue
    ds.add_container("class", param=1)  # Python keyword
    assert hasattr(ds, "class")  # Warn user about this in docs?
    # Suggest adding validation if needed:
    # with pytest.raises(ValueError):
    #     ds.add_container("invalid@name")

# --- Utility Function Tests ---
def test_build_tree_dict(dataset_storer):
    tree = _build_tree_dict(dataset_storer)
    assert "root" in tree
    assert "param1" in tree["root"]
    assert "child1" in tree["root"]
    assert "child2" in tree["root"]

# Edge Case: Empty DatasetStorer for tree dict
def test_build_tree_dict_empty():
    ds = DatasetStorer("empty")
    tree = _build_tree_dict(ds)
    assert tree == {"empty": {}}

def test_build_networkx_graph(dataset_storer):
    G = build_networkx_graph(dataset_storer, use_pydot=False)
    assert G.number_of_nodes() == 7
    assert G.number_of_edges() == 6
    assert G.has_edge("root", "param1")

# Edge Case: Empty DatasetStorer for graph
def test_build_networkx_graph_empty():
    ds = DatasetStorer("empty")
    G = build_networkx_graph(ds, use_pydot=False)
    assert G.number_of_nodes() == 1  # Just the root node
    assert G.number_of_edges() == 0

# Edge Case: Deeply nested structure
def test_build_networkx_graph_deep_nesting():
    ds = DatasetStorer("root")
    current = ds
    for i in range(5):  # 5 levels deep
        current = current.add_container(f"level{i}")
    G = build_networkx_graph(ds, use_pydot=False)
    assert G.number_of_nodes() == 6  # root + 5 levels
    assert G.number_of_edges() == 5