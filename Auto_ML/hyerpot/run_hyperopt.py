import os

import pandas as pd
from hpsklearn import HyperoptEstimator, random_forest, svc, knn
from hpsklearn import any_preprocessing
from hyperopt import hp
from hyperopt import tpe
from sklearn.model_selection import train_test_split

seed = 2022
data = pd.read_csv(os.path.join("./data", "wine.csv"))
# data = data.astype({'quality':'str'})

data = data[data.columns[:-1]]
X_cols, y_cols = data.columns[:-1], data.columns[-1]
trainset, testset = train_test_split(data, test_size=0.2, shuffle=False, random_state=seed)
X_train, y_train = trainset[X_cols], trainset[y_cols]
X_test, y_test = testset[X_cols], testset[y_cols]
# scaling data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

clf = hp.pchoice( 'my_name',
          [ ( 0.4, random_forest('my_name.random_forest') ),
            ( 0.3, svc('my_name.svc') ),
            ( 0.3, knn('my_name.knn') ) ])

# regularization candiate 정의
model = HyperoptEstimator(classifier=clf,
                          preprocessing=any_preprocessing('pre'),
                          algo=tpe.suggest,
                          max_evals=50,
                          trial_timeout=30)
model.fit(X_train, y_train)


acc = model.score(X_test, y_test)
print("Accuracy: %.3f" % acc)
# summarize the best model
print(model.best_model())
