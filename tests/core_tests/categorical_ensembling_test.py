import os
import sys

import numpy as np

import tests.utils_testing as utils
from auto_ml import Predictor

sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
sys.path = [os.path.abspath(os.path.dirname(os.path.dirname(__file__)))] + sys.path

os.environ['is_test_suite'] = 'True'


def test_categorical_ensemble_basic_classifier():
    np.random.seed(0)

    df_titanic_train, df_titanic_test = utils.titanic_binary_class_data()

    column_descriptions = {
        'survived': 'output'
        , 'pclass': 'categorical'
        , 'embarked': 'categorical'
        , 'sex': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='classifier',
                             column_descriptions=column_descriptions)

    ml_predictor.train_categorical_ensemble(df_titanic_train, categorical_column='pclass',
                                            optimize_final_model=False)

    test_score = ml_predictor.score(df_titanic_test, df_titanic_test.survived)

    print('test_score')
    print(test_score)

    # Small sample sizes mean there's a fair bit of noise here
    assert -0.155 < test_score < -0.135


def test_categorical_ensembling_regression(model_name=None):
    np.random.seed(0)

    df_boston_train, df_boston_test = utils.get_boston_regression_dataset()

    column_descriptions = {
        'MEDV': 'output'
        , 'CHAS': 'categorical'
    }

    ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

    ml_predictor.train_categorical_ensemble(df_boston_train, perform_feature_selection=True,
                                            model_names=model_name, categorical_column='CHAS')

    test_score = ml_predictor.score(df_boston_test, df_boston_test.MEDV)

    print('test_score')
    print(test_score)

    lower_bound = -4.2

    assert lower_bound < test_score < -2.8
