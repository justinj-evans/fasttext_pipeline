import pandas as pd
import numpy as np
import os
import io
import pickle
import progressbar
import collections
from collections import namedtuple
import re
import six
import unicodedata
from sklearn.model_selection import train_test_split

def create_train_test_validate():
    # load the arguements file
    with open("args.txt", "rb") as file:
        args = pickle.load(file)

    if args['train_filename'] or args['valid_filename'] or args['test_filename'] is "blank":

        primaryfile = pd.read_csv(args['filename_directory']+args['filename'], encoding=args['encoding'])

        traindata, test_validdata = train_test_split(primaryfile, test_size=0.3, stratify=None)
        validdata, testdata = train_test_split(test_validdata, test_size=0.5, stratify=None)
        print('training data size:',traindata.shape[0])
        print('validation data size:', testdata.shape[0])
        print('testing data size:', testdata.shape[0])

        traindata.to_csv(args['data_directory']+"traindata.csv")
        validdata.to_csv(args['data_directory'] + "validdata.csv")
        testdata.to_csv(args['data_directory'] + "testdata.csv")

    else:
        traindata = pd.read_csv(args['filename_directory'] + args['train_filename'], encoding=args['encoding'])
        validdata = pd.read_csv(args['filename_directory'] + args['valid_filename'], encoding=args['encoding'])
        testdata = pd.read_csv(args['filename_directory'] + args['test_filename'], encoding=args['encoding'])

        traindata.to_csv(args['data_directory']+ "traindata.csv")
        validdata.to_csv(args['data_directory'] + "validdata.csv")
        testdata.to_csv(args['data_directory'] + "testdata.csv")

def create_label_key(path, data):
    # load the arguements file
    with open("args.txt", "rb") as file:
        args = pickle.load(file)

    df = pd.read_csv(path+data)

    # create a numerical key
    df_code = df[[args['code']]]
    df_code.columns = ['code_text']
    code_key = df_code.groupby('code_text').size().reset_index()
    code_key['code'] = code_key.index + 1
    code_key['code'] = code_key['code'].astype(str).str.zfill(5)
    key_dict = pd.Series(code_key.code.values,index=code_key.code_text).to_dict()

    # save and load
    with open("code_key.txt", "wb") as file:
        pickle.dump(key_dict, file)

def strip_accents(text):
    try:
        text = unicodedata.normalize("NFD",text).encode("ascii","ignore").decode("utf-8")
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
print(cipher("this is an example", 2))

def preprocess_dataset(path, data):
    # load the arguements file
    with open("args.txt", "rb") as file:
        args = pickle.load(file)
    with open("code_key.txt", "rb") as file:
        code_dict = pickle.load(file)
        code_key = pd.DataFrame(code_dict.items(), columns=['code_text', 'code'])
    df = pd.read_csv(path+data)

    df = df.rename(columns={args['text_primary']: 'text_primary', args['text_supp1']: 'text_supp1',
                            args['text_supp2']: 'text_supp2',args['text_supp3']: 'text_supp3',
                            args['code']: 'code_text'})

    if args['text_supp1'] is "blank":
        df['code'] = df.code_text.map(code_dict)
        df["text_primary"] = df.apply(lambda row: strip_accents(row["text_primary"]), axis=1)
        df["text_primary"].apply(lambda x: x.lower())
        df["formatted"] = df.apply(lambda row: str('__label__' + row['code'] + ' ' + row['text_primary']), axis=1)
        np.savetxt(args['data_directory'] + "traindata_formatted.txt", df.formatted.values, fmt="%s")
        df.to_csv(args['data_directory'] + "traindata_formatted.csv")

    if args['text_primary'] and args['text_supp1'] is not "blank" and args['text_supp2'] is "blank":
        df['code'] = df.code_text.map(code_dict)
        df.apply(lambda x: x.astype(str).str.upper())
        df["text_primary"] = df.apply(lambda row: strip_accents(row["text_primary"]), axis=1)
        df["text_supp1"] = df.apply(lambda row: strip_accents(row["text_supp1"]), axis=1)

        df["formatted"] = df.apply(lambda row: str('__label__' + row['code'] + ' ' + row['text_primary'] +
                                                   cipher(row['text_supp1'],1)), axis=1)
        np.savetxt(args['data_directory'] + "traindata_formatted.txt", df.formatted.values, fmt="%s")
        df.to_csv(args['data_directory'] + "traindata_formatted.csv")

    if args['text_primary'] and args['text_supp1'] and args['text_supp2']  is not "blank" and args['text_supp3'] is "blank":
        df['code'] = df.code_text.map(code_dict)
        df.apply(lambda x: x.astype(str).str.upper())
        df["text_primary"] = df.apply(lambda row: strip_accents(row["text_primary"]), axis=1)
        df["text_supp1"] = df.apply(lambda row: strip_accents(row["text_supp1"]), axis=1)
        df["text_supp2"] = df.apply(lambda row: strip_accents(row["text_supp2"]), axis=1)

        df["formatted"] = df.apply(lambda row: str('__label__' + row['code'] + ' ' + row['text_primary'] +
                                                   cipher(row['text_supp1'],1) + ' ' +
                                                   cipher(row['text_supp2'],2)), axis=1)
        np.savetxt(args['data_directory'] + "traindata_formatted.txt", df.formatted.values, fmt="%s")
        df.to_csv(args['data_directory'] + "traindata_formatted.csv")

    if args['text_primary'] and args['text_supp1'] and args['text_supp2'] and args['text_supp3'] is not "blank":
        df['code'] = df.code_text.map(code_dict)
        df.apply(lambda x: x.astype(str).str.upper())
        df["text_primary"] = df.apply(lambda row: strip_accents(row["text_primary"]), axis=1)
        df["text_supp1"] = df.apply(lambda row: strip_accents(row["text_supp1"]), axis=1)
        df["text_supp2"] = df.apply(lambda row: strip_accents(row["text_supp2"]), axis=1)
        #df["text_supp3"] = df.apply(lambda row: strip_accents(row["text_supp3"]), axis=1)

        df["formatted"] = df.apply(lambda row: str('__label__' + row['code'] + ' ' + row['text_primary'] +
                                                   cipher(row['text_supp1'],1) + ' ' +
                                                   cipher(row['text_supp2'],2) + ' ' +
                                                   cipher(row['text_supp3'],3)), axis=1)
        np.savetxt(args['data_directory'] + "traindata_formatted.txt", df.formatted.values, fmt="%s")
        df.to_csv(args['data_directory'] + "traindata_formatted.csv")


# main preprocessing step
def run_preprocess():

    # load the arguements file
    with open("args.txt", "rb") as file:
        args = pickle.load(file)

    # create the three files
    create_train_test_validate()

    # start preprocessing
    create_label_key(args['data_directory'],"traindata.csv")

    # determine if cipher
    preprocess_dataset(args['data_directory'],"traindata.csv")