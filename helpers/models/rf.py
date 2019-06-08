from sklearn.ensemble import RandomForestClassifier


def rfc_train(mlcontext):
    rfc = RandomForestClassifier()
    rfc.fit(mlcontext.X_train, mlcontext.y_train)
    mlcontext.model = rfc
    return mlcontext
