from cash_ml.classifier import Classifier
from cash_ml.regressor import Regressor


class Predictor(object):

    def __init__(self, type_of_estimator, column_descriptions, verbose=True, name=None):
        self.training_features = None
        if type_of_estimator.lower() in [
            'regressor', 'regression', 'regressions', 'regressors', 'number', 'numeric',
            'continuous'
        ]:
            self.type_of_estimator = 'regressor'
        elif type_of_estimator.lower() in [
            'classifier', 'classification', 'categorizer', 'categorization', 'categories',
            'labels', 'labeled', 'label'
        ]:
            self.type_of_estimator = 'classifier'
        else:
            print('Invalid value for "type_of_estimator". Please pass in either "regressor" or '
                  '"classifier". You passed in: ' + type_of_estimator)
            raise ValueError(
                'Invalid value for "type_of_estimator". Please pass in either "regressor" or '
                '"classifier". You passed in: ' + type_of_estimator)
        if self.type_of_estimator == 'regressor':
            self.model = Regressor(
                type_of_estimator=type_of_estimator,
                column_descriptions=column_descriptions,
                verbose=verbose,
                name=name)
        else:
            self.model = Classifier(
                type_of_estimator=type_of_estimator,
                column_descriptions=column_descriptions,
                verbose=verbose,
                name=name)

    def train(self, *args, **kwargs):
        return self.model.train(*args, **kwargs)

    def score(self, *args, **kwargs):
        return self.model.score(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return self.model.predict_proba(*args, **kwargs)

    def predict_intervals(self, *args, **kwargs):
        return self.model.predict_intervals(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.model.save(*args, **kwargs)

    def train_categorical_ensemble(self, *args, **kwargs):
        return self.model.train_categorical_ensemble(*args, **kwargs)

    def transform_only(self, *args, **kwargs):
        return self.model.transform_only(*args, **kwargs)

    def predict_uncertainty(self, *args, **kwargs):
        return self.model.predict_uncertainty(*args, **kwargs)

    def score_uncertainty(self, *args, **kwargs):
        return self.model.score_uncertainty(*args, **kwargs)
