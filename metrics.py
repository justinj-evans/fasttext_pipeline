from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from matplotlib import pyplot
import pandas as pd
import pickle
import numpy as np

import mlflow
import mlflow.sklearn

# https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/

# load the arguements file
with open("args.txt", "rb") as file:
    args = pickle.load(file)

def load_data(path, data):
    df = pd.read_csv(path + data + ".csv")
    return df

def bootstrap(df):

    # run bootstrap
    n_iterations = int(args['n_iterations'])
    n_size = int(len(df)*float(args["n_size"]))
    stats = list()

    for i in range(n_iterations):
        boot = resample(df, n_samples=n_size)
        code = boot.code_text
        pred = boot.code_text_pred
        score = accuracy_score(code, pred)
        stats.append(score)

    # confidence intervals
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100))

    # track model performance
    mlflow.log_metric("bootstrap_cl_lower", lower)
    mlflow.log_metric("bootstrap_cl_upper", upper)
    mlflow.end_run()

def f1_precision_recall(df, name):
    weighted_f1 = round(f1_score(df.code_text, df.code_text_pred, average="weighted") * 100, 2)
    macro_f1 = round(f1_score(df.code_text, df.code_text_pred, average="macro") * 100, 2)
    weighted_precision = round(precision_score(df.code_text, df.code_text_pred, average="weighted") * 100, 2)
    macro_precision = round(precision_score(df.code_text, df.code_text_pred, average="macro") * 100, 2)
    weighted_recall = round(recall_score(df.code_text, df.code_text_pred, average="weighted") * 100, 2)
    macro_recall = round(recall_score(df.code_text, df.code_text_pred, average="macro") * 100, 2)

    #weighted_f1_name = name+"_weighted_f1"
    #mlflow.log_metric(weighted_f1_name, weighted_f1)

    print(name)
    print("weighted_f1: ", weighted_f1)
    print("weighted_precision: ", weighted_precision)
    print("weighted_recall: ", weighted_recall)

    # track performance
    mlflow.log_metric("weighted_f1", weighted_f1)
    mlflow.log_metric("macro_f1", macro_f1)
    mlflow.log_metric("weighted_precision", weighted_precision)
    mlflow.log_metric("macro_precision", macro_precision)
    mlflow.log_metric("weighted_recall", weighted_recall)
    mlflow.log_metric("macro_recall", macro_recall)

def run_metrics(path, data):
    df = load_data(path, data)

    f1_precision_recall(df, data)
    bootstrap(df)



