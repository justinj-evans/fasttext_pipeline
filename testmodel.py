import fasttext
import os
import pickle
import pandas as pd

# load the arguements file
with open("args.txt", "rb") as file:
    args = pickle.load(file)

def assign_prediction_score(path, data):
    with open("code_key.txt", "rb") as file:
        code_dict = pickle.load(file)
        inv_map = {v: k for k, v in code_dict.items()}

    loaded_model = fasttext.load_model(args['model_directory'] + args['model_name'] + ".bin")
    df = pd.read_csv(path + data + ".csv")

    pred_classes = []
    scores = []
    for line in df['preprocessed']:
        text = line.replace("__label__", "")
        pred = loaded_model.predict(text)
        pred_class = pred[0][0].replace("__label__","")
        pred_score = pred[1][0]
        pred_classes.append(pred_class)
        scores.append(pred_score)
    df['pred'] = pred_classes
    df['score'] = scores
    df['code_text_pred'] = df.pred.map(inv_map)

    df.to_csv(args['data_directory'] + data + "_predicted.csv")

# main train model step
def run_testmodel():
    assign_prediction_score(args['data_directory'], 'validdata_preprocessed')
    assign_prediction_score(args['data_directory'], 'testdata_preprocessed')





