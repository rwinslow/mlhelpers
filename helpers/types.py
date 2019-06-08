class MLContext(object):
    SCHEME = [
        "raw",
        "X",
        "y",
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "y_test_proba",
        "model",
    ]

    def __init__(self, **kwargs):
        for attr in self.SCHEME:
            setattr(self, attr, kwargs.get(attr, None))
