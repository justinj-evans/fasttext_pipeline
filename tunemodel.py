import sklearn.gaussian_process as gp
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import fasttext
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from config import update_arguements

# load the arguements file ## MOVE UP
with open("args.txt", "rb") as file:
    args = pickle.load(file)


def create_train_test(path, data):
    # raw_data = pd.read_csv(path + data, sep=" ", header=None)
    with open(path + data) as f:
        raw_data = f.readlines()
    raw_data = [x.strip() for x in raw_data]

    train, test = train_test_split(raw_data, test_size=0.3, stratify=None)

    np.savetxt(args['data_directory'] + "temp_train.txt", train, fmt="%s")
    np.savetxt(args['data_directory'] + "temp_test.txt", test, fmt="%s")


# code referenced from github.com/thuijskens/bayesian-optimization, YanPeng Gao

def get_tuning_params(bounds, hyper_params):
    discrete = ["min_char_grams", "max_char_grams", "word_ngrams", "epochs", "minimum_word_count"]
    params = []
    for index, item in enumerate(bounds):
        if hyper_params[index] in discrete:
            if hyper_params[index] == "min_char_grams" and "max_char_grams" in hyper_params and hyper_params.index(
                    "min_char_grams") > hyper_params.index("max_char_grams"):
                params.append(
                    np.random.randint(low=item[0], high=min(bounds[hyper_params.index("max_char_grams")][0], item[1]) + 1))
            elif hyper_params[index] == "max_char_grams" and "min_char_grams" in hyper_params and hyper_params.index(
                    "max_char_grams") > hyper_params.index("min_char_grams"):
                params.append(
                    np.random.randint(low=max(item[0], bounds[hyper_params.index("min_char_grams")][1]), high=item[1] + 1))
            else:
                params.append(np.random.randint(low=item[0], high=item[1] + 1))
        else:
            params.append(np.random.uniform(low=item[0], high=item[1]))

    return np.array(params)


def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, bounds, hyper_params,
                               greater_is_better=False, n_restarts=25):
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for i in range(n_restarts):
        starting_point = get_tuning_params(bounds=bounds, hyper_params=hyper_params)
        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))
        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x


def bayesian_optimisation(n_iters, sample_loss, tuning_vars, n_pre_samples=5, alpha=1e-5, epsilon=1e-7, saved_df=None):
    # tuning vars list of lists
    # bounds, np array (matrix)
    # hyper_params list of params
    temp_bounds_list = []
    hyper_params = []
    for index, item in enumerate(tuning_vars):
        hyper_params.append(item[0])
        temp_bounds_list.append(np.array([item[1], item[2]]))

    bounds = np.array(temp_bounds_list)

    x_list = []
    y_list = []

    n_params = bounds.shape[0]

    if saved_df is None:
        for i in range(n_pre_samples):
            params = get_tuning_params(bounds=bounds, hyper_params=hyper_params)
            x_list.append(params)
            y_list.append(sample_loss(params, hyper_params))
    else:
        numpy_matr = saved_df.values

        for i in range(numpy_matr.shape[0]):
            saved_row = numpy_matr[i][:-1].copy()
            saved_y = numpy_matr[i][-1].copy()
            x_list.append(saved_row)
            y_list.append(float(saved_y))

    xp = np.array(x_list)
    yp = np.array(y_list)
    matr = np.column_stack((xp, yp))
    col_names = hyper_params.copy()
    col_names.append(args["hyperparameter_metric"])
    df = pd.DataFrame(matr, columns=col_names)
    df.to_csv(args['model_directory'] + args['model_name'] + "_hyperparametertune_results.csv")

    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(kernel=kernel,
                                        alpha=alpha,
                                        n_restarts_optimizer=10,
                                        normalize_y=True)

    for n in range(n_iters):

        model.fit(xp, yp)

        # Sample next hyperparameter
        next_sample = sample_next_hyperparameter(acquisition_func=expected_improvement, gaussian_process=model,
                                                 evaluated_loss=yp, greater_is_better=True, bounds=bounds,
                                                 hyper_params=hyper_params, n_restarts=100)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = get_tuning_params(bounds=bounds, hyper_params=hyper_params)

        # Sample loss for new set of parameters
        cv_score = sample_loss(next_sample, hyper_params, n + 1)

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

        matr = np.column_stack((xp, yp))  # TODO fix bad practice here
        col_names = hyper_params.copy()
        col_names.append(args["hyperparameter_metric"])
        df = pd.DataFrame(matr, columns=col_names)
        df.to_csv(args['model_directory'] + args['model_name'] + "_hyperparametertune_results.csv")

    return xp, yp


def sample_loss(points, hyperparams, n=0):
    train_data = os.path.join(os.getenv("DATADIR", ""), args['data_directory'] + "temp_train.txt")
    #test_data = os.path.join(os.getenv("DATADIR", ""), args['data_directory'] + "temp_test.txt")

    # default values
    minimum_word_count = args['minimum_word_count']
    word_ngrams = args['word_ngrams']
    min_char_grams = args['min_char_grams']
    max_char_grams = args['max_char_grams']
    learning_rate = args['learning_rate']
    dimensions = args['dimensions']
    epoch = args['epochs']

    for i in range(len(hyperparams)):
        key = hyperparams[i]
        value = points[i]
        if key == "minimum_word_count":
            minimum_word_count = int(round(value))
        elif key == "word_ngrams":
            word_ngrams = int(round(value))
        elif key == "min_char_grams":
            min_char_grams = int(round(value))
        elif key == "max_char_grams":
            max_char_grams = int(round(value))
        elif key == "learning_rate":
            learning_rate = round(value, 2)
        elif key == "dimensions":
            dimensions = int(round(value))
        elif key == "epochs":
            epochs = int(round(value))
        else:
            raise Exception("Invalid Hyperparameter: ", key)

    model = fasttext.train_supervised(input=train_data, minCount=minimum_word_count,
                                      wordNgrams=word_ngrams, minn=min_char_grams, maxn=max_char_grams,
                                      lr=learning_rate, dim=dimensions, epoch=epochs)

    print("Iteration", n, "complete")

    # evaluate the model that was built
    with open(args['data_directory'] + "temp_test.txt") as file:
        df = file.readlines()

    labels = []
    pred_labels = []

    for line in df:
        label = line.split(' ', 1)[0].replace('__label__', '')
        labels.append(label)
        text = line.split(' ', 1)[1].strip()
        pred = model.predict(text)
        pred_label = pred[0][0].replace("__label__","")
        pred_labels.append(pred_label)

    if args["hyperparameter_metric"] == "accuracy":
        accuracy = accuracy_score(labels, pred_labels)
        print("Accuracy: ", round(accuracy,4)*100, "%")
        return accuracy

    elif args["hyperparameter_metric"] == "weighted_f1":
        weighted_f1 = f1_score(labels, pred_labels, average="weighted")
        print("Weighted F1: ", round(weighted_f1, 4) * 100, "%")
        return weighted_f1

    elif args["hyperparameter_metric"] == "weighted_precision":
        weighted_precision = precision_score(labels, pred_labels, average="weighted")
        print("Weighted Precision: ", round(weighted_precision, 4) * 100, "%")
        return weighted_precision

    elif args["hyperparameter_metric"] == "weighted_recall":
        weighted_recall = recall_score(labels, pred_labels, average="weighted")
        print("Weighted Recall: ", round(weighted_recall, 4) * 100, "%")
        return weighted_recall

    else:
        print("hyperparameter metric options: accuracy, weighted_f1, weighted_precision, weighted_recall")

def update_model_args():
    df = pd.read_csv(args['model_directory'] + args['model_name'] + "_hyperparametertune_results.csv")
    df_sort = df.sort_values(by=[args['hyperparameter_metric']], ascending=False)

    try:
        update_arguements({"learning_rate": df_sort.learning_rate.iloc[0]})
    except Exception:
        print('learning_rate: save error')

    try:
        update_arguements({"epochs": df_sort.epochs.iloc[0]})
    except Exception:
        print('epochs: save error')

    try:
        update_arguements({"dimensions": df_sort.dimensions.iloc[0]})
    except Exception:
        print('dimensions: save error')

    try:
        update_arguements({"minimum_word_count": df_sort.minimum_word_count.iloc[0]})
    except Exception:
        print('minimum_word_count: save error')

    try:
        update_arguements({"word_ngrams": df_sort.word_ngrams.iloc[0]})
    except Exception:
        print('word_ngrams: save error')

    try:
        update_arguements({"min_char_grams": df_sort.min_char_grams.iloc[0]})
    except Exception:
        print('min_char_grams: save error')

    try:
        update_arguements({"max_char_grams": df_sort.max_char_grams.iloc[0]})
    except Exception:
        print('max_char_grams: save error')

def run_tunemodel(tune_vars, metric):
    update_arguements({"hyperparameter_metric": metric}) # decide which evaluatin metric during tuning
    create_train_test(args['data_directory'], "traindata_preprocessed.txt") # define the training dataset
    xp, yp = bayesian_optimisation(n_iters=1, sample_loss=sample_loss, tuning_vars=tune_vars) # run the tuning
    update_model_args() # update the args dictionary based on tuning results
