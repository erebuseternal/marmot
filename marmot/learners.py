from modAL.models import ActiveLearner
from importlib import import_module
from sklearn.ensemble import RandomForestClassifier


def random_forest_learner_factory(X_training, y_training, query_method, **config):
    return ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=getattr(import_module('modAL.uncertainty'), query_method),
        X_training=X_training, y_training=y_training
    )