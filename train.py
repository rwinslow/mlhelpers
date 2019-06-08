from sklearn.model_selection import train_test_split


def split(kaggle):
    kaggle.X_train, kaggle.X_test, kaggle.y_train, kaggle.y_test = train_test_split(
        kaggle.X,
        kaggle.y,
        test_size=kaggle.test_split_size,
        random_state=kaggle.random_seed,
    )
    return kaggle
