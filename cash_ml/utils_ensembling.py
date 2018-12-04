import os

import numpy as np
import pandas as pd
import pathos
from sklearn.base import BaseEstimator, TransformerMixin


class Ensembler(BaseEstimator, TransformerMixin):

    def __init__(self,
                 ensemble_predictors,
                 type_of_estimator,
                 ensemble_method='average',
                 num_classes=None):
        self.ensemble_predictors = ensemble_predictors
        self.type_of_estimator = type_of_estimator
        self.ensemble_method = ensemble_method
        self.num_classes = num_classes

    # Get a dataframe that is all the predictions from all the sub-models
    # Note that we will get these predictions in parallel (relatively quick)

    # TODO: Simplify
    def get_all_predictions(self, X):

        def get_predictions_for_one_estimator(estimator, X):
            estimator_name = estimator.name

            if self.type_of_estimator == 'regressor':
                predictions = estimator.predict(X)
            else:
                # For classifiers
                predictions = list(estimator.predict_proba(X))
            return_obj = {estimator_name: predictions}
            return return_obj

        # Don't bother parallelizing if this is a single dictionary
        if X.shape[0] == 1:
            predictions_from_all_estimators = map(
                lambda predictor: get_predictions_for_one_estimator(predictor, X),
                self.ensemble_predictors)

        else:

            # Pathos doesn't like datasets beyond a certain size. So fall back on single,
            # non-parallel predictions instead.
            if os.environ.get('is_test_suite', False) == 'True':
                predictions_from_all_estimators = map(
                    lambda predictor: get_predictions_for_one_estimator(predictor, X),
                    self.ensemble_predictors)

            else:
                # Open a new multiprocessing pool
                pool = pathos.multiprocessing.ProcessPool()

                # Since we may have already closed the pool, try to restart it
                # TODO: Try to improve flow control
                try:
                    pool.restart()
                except AssertionError:
                    pass
                predictions_from_all_estimators = pool.map(
                    lambda predictor: get_predictions_for_one_estimator(predictor, X),
                    self.ensemble_predictors)

                # Once we have gotten all we need from the pool, close it so it's not taking up
                # unnecessary memory
                pool.close()
                # TODO: Try to improve flow control
                try:
                    pool.join()
                except AssertionError:
                    pass

        predictions_from_all_estimators = list(predictions_from_all_estimators)

        results = {}
        for result_dict in predictions_from_all_estimators:
            results.update(result_dict)

        # if this is a single row we are getting predictions from, just return a dictionary with
        # single values for all the predictions
        if X.shape[0] == 1:
            return results
        else:
            predictions_df = pd.DataFrame.from_dict(results, orient='columns')

            return predictions_df

    def fit(self, X, y):
        return self

    # Public API to get a single prediction from each row, where that single prediction is
    # somehow an ensemble of all our trained sub-predictors

    def predict(self, X):

        predictions = self.get_all_predictions(X)

        # If this is just a single dictionary we're getting predictions from:
        if X.shape[0] == 1:
            # predictions is just a dictionary where all the values are the predicted values from
            # one of our sub-predictors. we'll want that as a list
            predicted_values = list(predictions.values())
            if self.ensemble_method == 'median':
                return np.median(predicted_values)
            elif self.ensemble_method == 'average' \
                    or self.ensemble_method == 'mean' \
                    or self.ensemble_method == 'avg':
                return np.average(predicted_values)
            elif self.ensemble_method == 'max':
                return np.max(predicted_values)
            elif self.ensemble_method == 'min':
                return np.min(predicted_values)

        else:

            if self.ensemble_method == 'median':
                return predictions.apply(np.median, axis=1).values
            elif self.ensemble_method == 'average' or self.ensemble_method == 'mean' \
                    or self.ensemble_method == 'avg':
                return predictions.apply(np.average, axis=1).values
            elif self.ensemble_method == 'max':
                return predictions.apply(np.max, axis=1).values
            elif self.ensemble_method == 'min':
                return predictions.apply(np.min, axis=1).values

    def get_predictions_by_class(self, predictions):
        predictions_by_class = []
        for class_idx in range(self.num_classes):
            class_predictions = [pred[class_idx] for pred in predictions]
            predictions_by_class.append(class_predictions)

        return predictions_by_class

    def predict_proba(self, X):

        predictions = self.get_all_predictions(X)

        # If this is just a single dictionary we're getting predictions from:
        if X.shape[0] == 1:
            # predictions is just a dictionary where all the values are the predicted values from
            # one of our sub-predictors. we'll want that as a list
            predicted_values = list(predictions.values())
            predicted_values = self.get_predictions_by_class(predicted_values)

            if self.ensemble_method == 'median':
                return [np.median(class_predictions) for class_predictions in predicted_values]
            elif self.ensemble_method == 'average' or self.ensemble_method == 'mean' \
                    or self.ensemble_method == 'avg':
                return [np.average(class_predictions) for class_predictions in predicted_values]
            elif self.ensemble_method == 'max':
                return [np.max(class_predictions) for class_predictions in predicted_values]
            elif self.ensemble_method == 'min':
                return [np.min(class_predictions) for class_predictions in predicted_values]

        else:
            classed_predictions = predictions.apply(self.get_predictions_by_class, axis=1)

            if self.ensemble_method == 'median':
                return classed_predictions.apply(np.median, axis=1)
            elif self.ensemble_method == 'average' or self.ensemble_method == 'mean' \
                    or self.ensemble_method == 'avg':
                return classed_predictions.apply(np.average, axis=1)
            elif self.ensemble_method == 'max':
                return classed_predictions.apply(np.max, axis=1)
            elif self.ensemble_method == 'min':
                return classed_predictions.apply(np.min, axis=1)
