import pydot
import networkx as nx
import matplotlib.pyplot as plt

from ..data import DatasetStorer

from typing import *
from networkx import DiGraph

def _build_tree_dict(obj:DatasetStorer) -> Dict[str,Any]:
    """Recursively extracts the tree structure from a DatasetStorer instance, including non-container attributes."""
    tree = {obj.name: {}}
        
    for attr_name in dir(obj):
        if not attr_name.startswith("_") and (attr_name != 'name'):
            attr_value = getattr(obj, attr_name)
            if isinstance(attr_value, DatasetStorer):
                tree[obj.name][attr_name] = _build_tree_dict(attr_value)[attr_name]
            elif not callable(attr_value):
                tree[obj.name][attr_name] = None  # Representing parameters as leaf nodes

    return tree


def _display_tree(tree:Dict[str,Any], indent:int=0) -> None:
    """Displays the tree structure in a readable format."""
    for key, sub_tree in tree.items():
        print("  " * indent + "- " + key)
        if isinstance(sub_tree, dict):
            _display_tree(sub_tree, indent + 1)


def build_networkx_graph(obj: DatasetStorer, use_pydot:bool=False) -> DiGraph:
    """Builds a NetworkX graph from the DatasetStorer hierarchy."""
    if use_pydot:
        G = pydot.Dot(graph_type='graph')
    else:
        G = nx.DiGraph()
    
    def add_edges(parent_name:str, obj:DatasetStorer):
        for attr_name in dir(obj):
            if not attr_name.startswith("_") and (attr_name != 'name'):
                attr_value = getattr(obj, attr_name)
                if isinstance(attr_value, DatasetStorer):
                    if use_pydot:
                        G.add_edge(pydot.Edge(parent_name, attr_name))
                    else:
                        G.add_edge(parent_name, attr_name)
                    add_edges(attr_name, attr_value)
                elif not callable(attr_value):
                    if use_pydot:
                        G.add_edge(pydot.Edge(parent_name, attr_name))
                    else:
                        G.add_edge(parent_name, attr_name)  # Add parameters as leaf nodes
    
    if use_pydot:
        G.add_node(pydot.Node(obj.name))
    else:
        G.add_node(obj.name)
    add_edges(obj.name, obj)
    return G


def visualize_tree(obj: DatasetStorer, use_pydot:bool=False):
    """Visualizes the DatasetStorer hierarchy using NetworkX and Matplotlib."""
    G = build_networkx_graph(obj, use_pydot=use_pydot)
    
    if use_pydot:
        G = nx.nx_pydot.from_pydot(G)
        
    pos = nx.kamada_kawai_layout(G)
    # nx.spectral_layout
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
    plt.show()
    
def print_datasetstorer_hierarchy(obj:DatasetStorer) -> Dict[str,Any]:
    tree = _build_tree_dict(obj)
    _display_tree(tree)
    return tree