import sys

sys.path.insert(0, 'C:/Users/Justin Evans/Documents/Python/fasttext_project/')

import mlflow
from mlflow.tracking import MlflowClient
import pickle
from config import default_arguements, update_arguements
from preprocess import run_preprocess
from tunemodel import run_tunemodel
from trainmodel import run_trainmodel
from testmodel import run_testmodel
from metrics import run_metrics, setthreshold_testdata
import pandas as pd
from config import ml_finish_run

## start tracking ##
mlflow.set_experiment("model_1")
mlflow.start_run()

## load the arguements file ##
default_arguements()

## Define what the file looks like ##

update_arguements({"text_primary": "Plot", "code": "Genre",
                   "filename": "wiki_movie_plots_deduped.csv",
                   "filename_directory": "C:/Users/Justin Evans/Documents/Python/fasttext_project/input/",
                   'encoding': 'UTF-8',
                   "data_directory": "C:/Users/Justin Evans/Documents/Python/fasttext_project/data/"})

'''
update_arguements({"text_primary": "job_title", "code": "category",
                   "filename": "jobboard_reed_uk_primary.csv",
                   "filename_directory": "C:/Users/Justin Evans/Documents/Python/fasttext_project/input/",
                   'encoding': 'UTF-8',
                   "data_directory": "C:/Users/Justin Evans/Documents/Python/fasttext_project/data/"})

update_arguements({"text_primary": "Desc_E", "code": "Code_E", "text_supp1": "Desc_F", "text_supp2": "Desc_P",
                   "filename": "StatsPoland_ECOICOP.csv",
                   "filename_directory": "C:/Users/Justin Evans/Documents/Python/fasttext_project/input/",
                   'encoding': 'UTF-8',
                   "data_directory": "C:/Users/Justin Evans/Documents/Python/fasttext_project/data/"})
'''

## Preprocess the file ##
run_preprocess()

## Hyperparameter tune the model - bayesian_optimisation ##
update_arguements({"model_name": "model_1",
                   "model_description": "standard model, not hyperparameter-tuned",
                   "model_directory": "C:/Users/Justin Evans/Documents/Python/fasttext_project/model/"})

tune_variables = [["epochs", 10, 10], ["learning_rate", 0.1, 1], ["dimensions", 100, 100],
                  ["min_char_grams", 1, 4], ["max_char_grams", 1, 6],
                  ["minimum_word_count", 0, 8], ["word_ngrams", 1, 5]]

#run_tunemodel(tune_variables, "accuracy")

## Train the model ##

# note - don't define the model parameters here if you have hyperparameter tuned in the previous step
update_arguements({"model_quantize": "yes",
                   "epochs": "10", "learning_rate": "0.7", "dimensions": "60", "minimum_word_count": "1",
                   "word_ngrams": "6", "min_char_grams": "0", "max_char_grams": "5"})

run_trainmodel()

## Make model predictions ##
run_testmodel()

## Generate metrics ##
update_arguements({"metrics_directory": "C:/Users/Justin Evans/Documents/Python/fasttext_project/metrics/"})
update_arguements({"n_iterations": "100", "n_size": "0.5"})
with open("args.txt", "rb") as file:
    args = pickle.load(file)

# generate metrics: F1, precision, recall, bootstrapped accuracy
run_metrics(args['data_directory'], 'validdata_preprocessed_predicted')
run_metrics(args['data_directory'], 'testdata_preprocessed_predicted') # run test second to log results

# based on the valid data error rate & threshold, what is the error rate if applied to the test dataset
setthreshold_testdata(args['data_directory'], 'validdata_preprocessed_predicted', 'testdata_preprocessed_predicted')


## Tracking of ml iterations through ml flow ##
ml_finish_run()
