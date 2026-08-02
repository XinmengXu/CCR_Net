"""Model registry."""

from .CCRNet import CCRNet, CCRConfig, CCRStageReport

__all__ = ["CCRNet", "CCRConfig", "CCRStageReport"]


def register_model(custom_model):
    if custom_model.__name__ in globals() or custom_model.__name__.lower() in {
        key.lower() for key in globals()
    }:
        raise ValueError(f"Model {custom_model.__name__} already exists.")
    globals()[custom_model.__name__] = custom_model


def get(identifier):
    if not isinstance(identifier, str):
        raise ValueError(f"Could not interpret model name: {identifier}")
    models = {key.lower(): value for key, value in globals().items()}
    model_class = models.get(identifier.lower())
    if model_class is None:
        raise ValueError(f"Could not interpret model name: {identifier}")
    return model_class
