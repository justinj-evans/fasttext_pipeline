import pandas as pd
import numpy as np
import os
import pickle

# set up arguements
def arguements(updates):
    """Data Preprocessing Options"""

    # create dictionary
    arg_dict = dict(text_primary="",
                    code="",
                    text_supp1="blank",
                    text_supp2="blank",
                    text_supp3="blank",
                    filename_directory="",
                    filename="",
                    tok="True",
                    data_directory="",
                    train_filename="blank",
                    valid_filename="blank",
                    test_filename="blank",
                    vocab_size="80000",
                    encoding="utf-8")

    arg_dict.update(updates)

    # save and load
    with open("args.txt", "wb") as file:
        pickle.dump(arg_dict, file)

    with open("args.txt", "rb") as file:
        arg_dict = pickle.load(file)

    return (print(arg_dict))
