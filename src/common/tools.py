import pickle

import yaml


def load_config():
    """Load configuration from config.yaml."""
    with open("config.yaml") as p:
        config = yaml.safe_load(p)
    return config


def pickle_dump(path, variable):
    """Serialize object to file."""
    with open(path, "wb") as handle:
        pickle.dump(variable, handle)


def pickle_load(path):
    """Load serialized object from file."""
    with open(path, "rb") as handle:
        loaded = pickle.load(handle)
    return loaded
