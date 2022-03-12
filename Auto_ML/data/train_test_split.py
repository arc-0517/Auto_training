import pandas as pd
import os
from sklearn.model_selection import train_test_split

seed = 2022
data = pd.read_csv(os.path.join("./data", "wine.csv"))

data.quliaty -= 3

data = data[data.columns[:-1]]

X_cols, y_cols = data.columns[:-1], data.columns[-1]
trainset, testset = train_test_split(data, test_size=0.2, shuffle=False, random_state=seed)

trainset.to_csv(os.path.join("./data", "wine_train.csv"), index=False)
testset.to_csv(os.path.join("./data", "wine_test.csv"), index=False)
