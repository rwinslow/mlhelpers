class KaggleContext(object):
    SCHEME = [
        # Raw data
        "raw",
        "X",
        "y",
        # Data pointer
        "data",
        # Train
        "train",
        "X_train",
        "y_train",
        # Test
        "test",
        "X_test",
        "y_test",
        "y_test_proba",
        # Model
        "model",
        # Controls
        "target_col",
        "label_col",
        "test_split_size",
        "random_seed",
        "min_max_scaler",
        "n_components",
    ]

    def __init__(self, **kwargs):
        for attr in self.SCHEME:
            setattr(self, attr, kwargs.get(attr, None))
