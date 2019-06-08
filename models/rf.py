from sklearn.ensemble import RandomForestClassifier


def rfc_train(kaggle):
    rfc = RandomForestClassifier()
    rfc.fit(kaggle.X_train, kaggle.y_train)
    kaggle.model = rfc
    return kaggle
