
# reference - https://www.kaggle.com/rishirajacharya/mmlm22-men-s-tuned
from autolgbm import AutoLGBM

# required parameters:
train_filename = "./data/train.csv"
output = "output"

# optional parameters
test_filename = "./data/test.csv"
task = "classification"     # task: classification or regression
idx = "ID"      # an id column. If not specified, the id column will be generated automatically with the name "id"
targets = ["Pred"]      # target columns are list of strings, multi output possible
features = None     # features columns are list of strings. if not specified, all columns except `id`, `targets` & `kfold` columns will be used
categorical_features = None     # categorical features are list of strings. if not specified, categorical columns will be inferred automatically.
use_gpu = False     # use_gpu is boolean
num_folds = 5      # number of folds to use for cross-validation. default is 5
seed = 2022
num_trials = 100      # number of optuna trials to run. default is 1000
time_limit = 360        # time_limit for optuna trials in seconds. if not specified, timeout is not set and all trials are run.
fast = False        # if fast is set to True, the hyperparameter tuning will use only one fold. however, the model will be trained on all folds in the end

# Now its time to train the model!
algbm = AutoLGBM(
    train_filename=train_filename,
    output=output,
    test_filename=test_filename,
    task=task,
    idx=idx,
    targets=targets,
    features=features,
    categorical_features=categorical_features,
    use_gpu=use_gpu,
    num_folds=num_folds,
    seed=seed,
    num_trials=num_trials,
    time_limit=time_limit,
    fast=fast,
)

algbm.train()

'''
submission = pd.read_csv("./output2/test_predictions.csv")
submission.drop('0.0', inplace=True, axis=1)
submission.rename(columns = {'1.0':'Pred'}, inplace = True)
submission.to_csv("submission.csv", index=False)
'''