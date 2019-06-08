from sklearn.linear_model import LogisticRegression


def logit_train(kaggle):
    logit = LogisticRegression(
        solver="liblinear", penalty="l2", C=0.1, class_weight="balanced"
    )
    logit.fit(kaggle.X_train, kaggle.y_train)
    kaggle.model = logit
    return kaggle
