import tpot
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 2022
data = pd.read_csv(os.path.join("../data", "wine.csv"))

data = data[data.columns[:-1]]
X_cols, y_cols = data.columns[:-1], data.columns[-1]
trainset, testset = train_test_split(data, test_size=0.2, shuffle=False, random_state=seed)
X_train, y_train = trainset[X_cols], trainset[y_cols]
X_test, y_test = testset[X_cols], testset[y_cols]
# scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pipeline_optimizer = tpot.TPOTClassifier(generations=5, #number of iterations to run the training
                                         population_size=20, #number of individuals to train
                                         cv=5,
                                         scoring='accuracy',
                                         random_state=seed,
                                         n_jobs=-1) #number of folds in StratifiedKFold

pipeline_optimizer.fit(X_train, y_train) #fit the pipeline optimizer - can take a long time

print(pipeline_optimizer.score(X_test, y_test)) #print scoring for the pipeline

pipeline_optimizer.export('./tpot_exported_pipeline_scaler.py') #export the pipeline

