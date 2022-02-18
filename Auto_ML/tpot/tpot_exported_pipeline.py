import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import ZeroCount
from xgboost import XGBClassifier
from tpot.export_utils import set_param_recursive
from sklearn.metrics import accuracy_score

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv(os.path.join("../data", "wine.csv"))
tpot_data = tpot_data[tpot_data.columns[:-1]]

features = tpot_data.drop('quality', axis=1)
training_features, testing_features, training_target, testing_target = train_test_split(features, tpot_data['quality'], random_state=2022)

exported_pipeline = make_pipeline(
    ZeroCount(),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    XGBClassifier(learning_rate=0.1, max_depth=8, min_child_weight=12, n_estimators=100, n_jobs=1, subsample=0.5, verbosity=0))

# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 2022)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
accuracy_score(testing_target, results)
