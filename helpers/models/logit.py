from sklearn.linear_model import LogisticRegression


def logit_train(mlcontext):
    logit = LogisticRegression(
        solver="liblinear", penalty="l2", C=0.1, class_weight="balanced"
    )
    logit.fit(mlcontext.X_train, mlcontext.y_train)
    mlcontext.model = logit
    return mlcontext
