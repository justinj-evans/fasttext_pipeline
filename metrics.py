from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
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
    n_size = int(len(df) * float(args["n_size"]))
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
    accuracy = round(accuracy_score(df.code_text, df.code_text_pred) * 100, 2)

    # weighted_f1_name = name+"_weighted_f1"
    # mlflow.log_metric(weighted_f1_name, weighted_f1)

    print(name)
    print("Accuracy: ", accuracy)
    print("Weighted_f1: ", weighted_f1)
    print("Weighted_precision: ", weighted_precision)
    print("Weighted_recall: ", weighted_recall)

    # track performance
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("weighted_f1", weighted_f1)
    mlflow.log_metric("macro_f1", macro_f1)
    mlflow.log_metric("weighted_precision", weighted_precision)
    mlflow.log_metric("macro_precision", macro_precision)
    mlflow.log_metric("weighted_recall", weighted_recall)
    mlflow.log_metric("macro_recall", macro_recall)


def setthreshold_testdata(path, validdata, testdata):
    # based on the valid dataset you define a threshold around 5% error, then see what value you get for the test
    # very messy work, will clean up
    range = np.arange(0.5, 1.0, 0.01).tolist()
    range.sort(reverse=True)
    df_scores = pd.DataFrame(range, columns=['confidence'])

    def threshold_error(data, value):
        df_temp = data[data['score'] > value]
        accuracy = round(accuracy_score(df_temp.code_text, df_temp.code_text_pred) * 100, 2)
        error_rate = 100 - accuracy
        return error_rate

    def threshold_percent(data, value):
        count_overall = data.shape[0]
        df_temp = data[data['score'] > value]
        count_threshold = df_temp.shape[0]
        percent = round((count_threshold / count_overall) * 100, 2)
        return percent

    df_valid = load_data(path, validdata)
    df_scores["error_rate"] = df_scores.apply(lambda row: threshold_error(df_valid, row['confidence']), axis=1)
    df_scores.confidence = df_scores.confidence * 10

    error_tolerance = 0.05
    df_selected = df_scores.iloc[(df_scores['error_rate'] - error_tolerance).abs().argsort()[:2]]
    threshold_selected = float(df_selected.confidence.iloc[0] / 10)

    # test data
    df_test = load_data(path, testdata)

    df_scores = pd.DataFrame(range, columns=['confidence'])
    df_scores["error_rate"] = df_scores.apply(lambda row: threshold_error(df_test, row['confidence']), axis=1)
    df_scores["autocoding_rate"] = df_scores.apply(lambda row: threshold_percent(df_test, row['confidence']), axis=1)

    # select row based on valid data threshold
    df_score_selected = df_scores[df_scores['confidence'] == threshold_selected]
    test_error = df_score_selected.error_rate.iloc[0]
    test_rate = df_score_selected.autocoding_rate.iloc[0]
    print("Based on the valid data error (near 5%), threshold selected:", round(threshold_selected, 2) * 10,"."
          "Selected threshold applied to test dataset: ", test_error, "% error rate,", test_rate, "% autocoding rate." )

    # export threshold
    df_scores.confidence = df_scores.confidence * 10
    df_scores.to_csv(args['metrics_directory'] + args['model_name'] + "_threshold_metrics.csv")

    # track performance
    mlflow.log_metric("validthreshold_testerror", test_error)


def run_metrics(path, data):
    df = load_data(path, data)

    f1_precision_recall(df, data)
    bootstrap(df)

    # track performance
    path = args['metrics_directory']
    mlflow.log_artifact(path)
