import sys
sys.path.insert(0, 'C:/Users/Justin Evans/Documents/Python/fasttext/')

from config import default_arguements, update_arguements
from preprocess import run_preprocess
from trainmodel import run_trainmodel
from testmodel import run_testmodel

# Define what the file looks like
default_arguements()
update_arguements({"text_primary": "Desc_E", "code": "Code_E", "text_supp1": "Desc_F", "text_supp2": "Desc_P",
            "filename": "StatsPoland_ECOICOP.csv",
            "filename_directory": "C:/Users/Justin Evans/Documents/Python/fasttext/input/", 'encoding': 'UTF-8',
            "data_directory": "C:/Users/Justin Evans/Documents/Python/fasttext/data/"})

# Preprocess the file
run_preprocess()

# Train the model
update_arguements({"model_name": "model_1", "model_quantize": "yes", "model_directory": "C:/Users/Justin Evans/Documents/Python/fasttext/model/",
            "epochs": "10", "learning_rate": "0.7", "dimensions": "60", "minimum_word_count": "6",
            "wordNgrams": "6", "min_char_grams": "4", "max_char_grams": "5"})
run_trainmodel()

# Make model predictions
run_testmodel()
