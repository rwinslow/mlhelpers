from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score


def classifier_score(kaggle):
    y_proba = kaggle.model.predict_proba(kaggle.X_test)
    y_proba = [v[1] for v in y_proba]
    print("ROC AUC Score: ", roc_auc_score(kaggle.y_test, y_proba))
    return kaggle


def get_residuals(kaggle):
    residuals = []
    for truth, predict in zip(kaggle.y_test, kaggle.model.predict(kaggle.X_test)):
        residuals.append({"truth": truth, "predict": predict, "diff": truth - predict})
    return residuals


def cross_validate(kaggle, scoring=None, cv=5):
    """Cross validate the model with some number of folds.

    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html

    Args:
        kaggle (Kaggle): Kaggle object populated with data for conducting cross-
            validation. Attributes used:
                X: Train data
                y: Target data
                model: Model to cross validate
        scoring (multi): A single string or a callable to evaluate the predictions on
            the test set. For evaluating multiple metrics, either give a list of
            (unique) strings or a dict with names as keys and callables as values.
        cv (int or iter): Number of folds or cross validation iterator.
    Returns:
        scores (list of float): Cross-validation scores for each fold.
    """
    scores = cross_val_score(kaggle.model, kaggle.X, kaggle.y, scoring=scoring, cv=cv)
    print(f"Accuracy (95% conf. int.): {scores.mean():0.2f} (+/- {scores.std()*2})")
    return scores


def classification_stats(kaggle, source="all"):
    """Get classification stats for a model.

    Args:
        kaggle (Kaggle): Kaggle object to analyze. Attributes used:
            model: Model to check.
        source (str): Determines which data to use from the Kaggle object. Options:
            all: Uses attributes X and y
            train: Uses attributes X_train and y_train
            test: Uses attributes X_test and y_test
    Returns:
        kaggle (Kaggle): Kaggle object after analysis.
    """
    if source == "all":
        X, y = kaggle.X, kaggle.y
    elif source == "train":
        X, y = kaggle.X_train, kaggle.y_train
    elif source == "test":
        X, y = kaggle.X_test, kaggle.y_test
    else:
        print("Invalid source. Skipping analysis.")
        return kaggle

    predicted = kaggle.model.predict(X)
    probabilities = kaggle.model.predict_proba(X)
    classes = [int(v) for v in list(set(y))]

    # Get Accuracy and ROC AUC results for each class individually
    start = 0
    if probabilities.shape[1] == 2:
        start = 1
    for i in range(start, probabilities.shape[1]):
        probs = probabilities[:, i]
        current_class = classes[i]
        y_test_i = [1 if current_class == int(v) else 0 for v in y]
        predicted_i = [1 if current_class == int(v) else 0 for v in predicted]
        print("Class {}".format(current_class))
        print("Accuracy: {:0.2f}".format(accuracy_score(y_test_i, predicted_i)))
        print("ROC AUC Score: {:0.2f}".format(roc_auc_score(y_test_i, probs)))
        print()

    print("Confusion Matrix")

    # Print out confusion matrix legend  if only 2 classes
    if len(classes) == 2:
        print("True Negative (Guess 0, Actual 0)  | False Positive (Guess 1, Actual 0)")
        print("-----------------------------------------------------------------------")
        print("False Negative (Guess 0, Actual 1) |  True Positive (Guess 1, Actual 1)")
        print()

    print(confusion_matrix(y, predicted))
    print()
    print("Classification Report")
    print(classification_report(y, predicted))

    return kaggle
