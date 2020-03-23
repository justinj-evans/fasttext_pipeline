import pandas as pd
import numpy as np
import pickle
import progressbar
import unicodedata
from sklearn.model_selection import train_test_split
import mlflow
import os

# load the arguements file
with open("args.txt", "rb") as file:
    args = pickle.load(file)


def create_train_test_validate():
    if args['train_filename'] or args['valid_filename'] or args['test_filename'] is "blank":

        primaryfile = pd.read_csv(args['filename_directory'] + args['filename'], encoding=args['encoding'])

        traindata, test_validdata = train_test_split(primaryfile, test_size=0.3, stratify=None)
        validdata, testdata = train_test_split(test_validdata, test_size=0.5, stratify=None)
        print('training data size:', traindata.shape[0])
        print('validation data size:', testdata.shape[0])
        print('testing data size:', testdata.shape[0])

        traindata.to_csv(args['data_directory'] + "traindata.csv")
        validdata.to_csv(args['data_directory'] + "validdata.csv")
        testdata.to_csv(args['data_directory'] + "testdata.csv")

    else:
        traindata = pd.read_csv(args['filename_directory'] + args['train_filename'], encoding=args['encoding'])
        validdata = pd.read_csv(args['filename_directory'] + args['valid_filename'], encoding=args['encoding'])
        testdata = pd.read_csv(args['filename_directory'] + args['test_filename'], encoding=args['encoding'])

        traindata.to_csv(args['data_directory'] + "traindata.csv")
        validdata.to_csv(args['data_directory'] + "validdata.csv")
        testdata.to_csv(args['data_directory'] + "testdata.csv")


def create_label_key(path, data):
    df = pd.read_csv(path + data + ".csv")

    # create a numerical key
    df_code = df[[args['code']]]
    df_code.columns = ['code_text']
    code_key = df_code.groupby('code_text').size().reset_index()
    code_key['code'] = code_key.index + 1
    code_key['code'] = code_key['code'].astype(str).str.zfill(5)
    key_dict = pd.Series(code_key.code.values, index=code_key.code_text).to_dict()

    # save and load
    with open("code_key.txt", "wb") as file:
        pickle.dump(key_dict, file)


def strip_accents(text):
    try:
        text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
        return text
    except Exception:
        return text


def cipher(sentence, shift_value):
    translated = ""  # new empty string of soon to be transformed string
    if sentence == "":
        return ""
    for char in sentence:
        if char.isalpha():
            ascii_num = ord(char)  # get the ascii rep of the character
            ascii_num += shift_value  # add the shifted value
            # handle index rapping
            if char.isupper():
                if ascii_num > ord("Z"):
                    ascii_num -= 26
                elif ascii_num < ord("A"):
                    ascii_num += 26
            elif char.islower():
                if ascii_num > ord("z"):
                    ascii_num -= 26
                elif ascii_num < ord("a"):
                    ascii_num += 26
            translated += chr(ascii_num)
        else:
            translated += char
    return str(translated)

def preprocess_dataset(path, data):
    with open("code_key.txt", "rb") as file:
        code_dict = pickle.load(file)
    raw_data = pd.read_csv(path + data + ".csv")

    print('hello world')

    df = raw_data.rename(columns={args['text_primary']: 'text_primary', args['text_supp1']: 'text_supp1',
                            args['text_supp2']: 'text_supp2', args['text_supp3']: 'text_supp3',
                            args['code']: 'code_text'})

    df['code'] = df.code_text.map(code_dict)
    for col in df:  # lowercase the dataframe
        if col == str:
            df[col] = df[col].apply(lambda x: x.lower())

    if args['text_supp1'] == "blank":
        df["text_primary"] = df.apply(lambda row: strip_accents(row["text_primary"]), axis=1)
        df["preprocessed"] = df.apply(lambda row: str('__label__' + row['code'] + ' ' + row['text_primary']), axis=1)
        np.savetxt(args['data_directory'] + data + "_preprocessed.txt", df.preprocessed.values, fmt="%s")
        df.to_csv(args['data_directory'] + data + "_preprocessed.csv")

    if args['text_primary'] and args['text_supp1'] != "blank" and args['text_supp2'] == "blank":
        df["text_primary"] = df.apply(lambda row: strip_accents(row["text_primary"]), axis=1)
        df["text_supp1"] = df.apply(lambda row: strip_accents(row["text_supp1"]), axis=1)

        df["preprocessed"] = df.apply(lambda row: str('__label__' + row['code'] + ' ' + row['text_primary'] +
                                                      cipher(row['text_supp1'], 1)), axis=1)
        np.savetxt(args['data_directory'] + data + "_preprocessed.txt", df.preprocessed.values, fmt="%s")
        df.to_csv(args['data_directory'] + data + "_preprocessed.csv")

    if args['text_primary'] and args['text_supp1'] and args['text_supp2'] != "blank" and args['text_supp3'] == "blank":
        df["text_primary"] = df.apply(lambda row: strip_accents(row["text_primary"]), axis=1)
        df["text_supp1"] = df.apply(lambda row: strip_accents(row["text_supp1"]), axis=1)
        df["text_supp2"] = df.apply(lambda row: strip_accents(row["text_supp2"]), axis=1)

        df["preprocessed"] = df.apply(lambda row: str('__label__' + row['code'] + ' ' + row['text_primary'] +
                                                      cipher(row['text_supp1'], 1) + ' ' +
                                                      cipher(row['text_supp2'], 2)), axis=1)
        np.savetxt(args['data_directory'] + data + "_preprocessed.txt", df.preprocessed.values, fmt="%s")
        df.to_csv(args['data_directory'] + data + "_preprocessed.csv")

    if args['text_primary'] and args['text_supp1'] and args['text_supp2'] and args['text_supp3'] != 'blank':
        df["text_primary"] = df.apply(lambda row: strip_accents(row["text_primary"]), axis=1)
        df["text_supp1"] = df.apply(lambda row: strip_accents(row["text_supp1"]), axis=1)
        df["text_supp2"] = df.apply(lambda row: strip_accents(row["text_supp2"]), axis=1)
        df["text_supp3"] = df.apply(lambda row: strip_accents(row["text_supp3"]), axis=1)

        df["preprocessed"] = df.apply(lambda row: str('__label__' + row['code'] + ' ' + row['text_primary'] +
                                                      cipher(row['text_supp1'], 1) + ' ' +
                                                      cipher(row['text_supp2'], 2) + ' ' +
                                                      cipher(row['text_supp3'], 3)), axis=1)
        np.savetxt(args['data_directory'] + data + "_preprocessed.txt", df.preprocessed.values, fmt="%s")
        df.to_csv(args['data_directory'] + data + "_preprocessed.csv")

    # log training data in mlflow

# main preprocessing step
def run_preprocess():
    # create the three files
    create_train_test_validate()

    # create a label for each class
    create_label_key(args['data_directory'], "traindata")

    # preprocess and cipher if necessary, export data
    preprocess_dataset(args['data_directory'], "traindata")
    preprocess_dataset(args['data_directory'], "validdata")
    preprocess_dataset(args['data_directory'], "testdata")

    print('hello world')
