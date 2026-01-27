import pickle

import yaml


def load_config():
    """
    Load configuration from config.yaml file.
    
    Returns:
        Dictionary containing configuration parameters
    """
    with open("config.yaml") as p:
        config = yaml.safe_load(p)
    return config


def pickle_dump(path, variable):
    """
    Serialize and save a Python object to a file.
    
    Args:
        path: File path to save the pickled object
        variable: Python object to serialize
    """
    with open(path, "wb") as handle:
        pickle.dump(variable, handle)


def pickle_load(path):
    """
    Load and deserialize a Python object from a pickle file.
    
    Args:
        path: File path to the pickled object
    
    Returns:
        Deserialized Python object
    """
    with open(path, "rb") as handle:
        loaded = pickle.load(handle)
    return loaded
