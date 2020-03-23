import sys

sys.path.insert(0, 'C:/Users/Justin Evans/Documents/Python/fasttext/')
import mlflow
from mlflow.tracking import MlflowClient
import pickle
from config import default_arguements, update_arguements
from preprocess import run_preprocess
from trainmodel import run_trainmodel
from testmodel import run_testmodel
from metrics import bootstrap
import pandas as pd

# start tracking
mlflow.set_experiment("model_1")
mlflow.start_run()

## load the arguements file ##
default_arguements()

## Define what the file looks like ##
update_arguements({"text_primary": "Desc_E", "code": "Code_E", "text_supp1": "Desc_F", "text_supp2": "Desc_P",
                   "filename": "StatsPoland_ECOICOP.csv",
                   "filename_directory": "C:/Users/Justin Evans/Documents/Python/fasttext_project/input/",
                   'encoding': 'UTF-8',
                   "data_directory": "C:/Users/Justin Evans/Documents/Python/fasttext_project/data/"})

## Preprocess the file ##
run_preprocess()

## Train the model ##
update_arguements({"model_name": "model_1", "model_quantize": "yes",
                   "model_directory": "C:/Users/Justin Evans/Documents/Python/fasttext_project/model/",
                   "epochs": "10", "learning_rate": "0.7", "dimensions": "60", "minimum_word_count": "6",
                   "wordNgrams": "6", "min_char_grams": "4", "max_char_grams": "5"})


#run_trainmodel()

#from trainmodel import track_test
#track_test()

# Make model predictions
# run_testmodel()

# Generate metrics
update_arguements({"n_iterations": "100", "n_size": "0.5"})
with open("args.txt", "rb") as file:
    args = pickle.load(file)


# bootstrap(args['data_directory'], 'validdata_preprocessed_predicted')


# Tracking of ml iterations through ml flow
def finish_tracking():
    mlflow.end_run()
    runs = pd.DataFrame(mlflow.search_runs())
    #print(runs)
    runs.to_csv("C:/Users/Justin Evans/Documents/Python/fasttext_project/run_scripts/mlruns/mlflow.csv")
finish_tracking()
