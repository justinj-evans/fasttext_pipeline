import _pickle as pickle
import sys
sys.path.insert(0, 'C:/Users/Justin Evans/Documents/Python/fasttext/')
#sys.path.append('C:/Users/Justin Evans/Documents/Python/fasttext/')

import config
from config import arguements

# define what the file looks like
arguements({"text_primary": "Desc_E", 'code': "Code_E", "text_supp1":"Desc_F","text_supp2":"Desc_P",
            "filename":"StatsPoland_ECOICOP.csv","filename_directory":"C:/Users/Justin Evans/Documents/Python/fasttext/input/",'encoding':'UTF-8',
            "data_directory":"C:/Users/Justin Evans/Documents/Python/fasttext/data/"})

# preprocess the file
from preprocess import run_preprocess

run_preprocess()







