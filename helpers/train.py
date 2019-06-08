from sklearn.model_selection import train_test_split


def split(mlcontext):
    (
        mlcontext.X_train,
        mlcontext.X_test,
        mlcontext.y_train,
        mlcontext.y_test,
    ) = train_test_split(
        mlcontext.X,
        mlcontext.y,
        test_size=mlcontext.test_split_size,
        random_state=mlcontext.random_seed,
    )
    return mlcontext
