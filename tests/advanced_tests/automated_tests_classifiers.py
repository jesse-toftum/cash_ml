import os
import sys
from collections import OrderedDict

import tests.advanced_tests.classifiers as classifier_tests

sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
os.environ['is_test_suite'] = 'True'

training_parameters = {
    'model_names': ['DeepLearning', 'GradientBoosting', 'XGB', 'LGBM', 'CatBoost']
}

# Make this an OrderedDict so that we run the tests in a consistent order
test_names = OrderedDict([
    ('getting_single_predictions_multilabel_classification',
     classifier_tests.getting_single_predictions_multilabel_classification),
    ('optimize_final_model_classification', classifier_tests.optimize_final_model_classification)
])


def test_generator():
    for model_name in training_parameters['model_names']:
        for test_name, test in test_names.items():
            test_model_name = model_name + 'Classifier'
            # test_model_name = model_name

            test.description = str(test_model_name) + '_' + test_name
            yield test, test_model_name
