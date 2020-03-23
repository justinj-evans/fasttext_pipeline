import pandas as pd
import numpy as np
import os
import pickle


# set up arguements
def default_arguements():
    # create dictionary
    arg_dict = dict(text_primary="",
                    code="",
                    text_supp1="blank",
                    text_supp2="blank",
                    text_supp3="blank",
                    filename_directory="",
                    filename="",

                    data_directory="",
                    train_filename="blank",
                    valid_filename="blank",
                    test_filename="blank",
                    vocab_size="80000",
                    encoding="utf-8",

                    model_directory="",
                    model_name="model_1",
                    model_quantize="yes",
                    epochs="20",
                    learning_rate="0.7",
                    dimensions="60",
                    minimum_word_count="6",
                    word_ngrams="6",
                    min_char_grams="4",
                    max_char_grams="5",

                    n_iterations="100",
                    n_size="0.50"

                    )

    # save and load
    with open("args.txt", "wb") as file:
        pickle.dump(arg_dict, file)

    return (print(arg_dict))


def update_arguements(updates):
    """Data Preprocessing Options"""
    with open("args.txt", "rb") as file:
        arg_dict = pickle.load(file)

    arg_dict.update(updates)

    # save and load
    with open("args.txt", "wb") as file:
        pickle.dump(arg_dict, file)
