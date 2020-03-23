import fasttext
import os
import pickle
import mlflow

# load the arguements file
with open("args.txt", "rb") as file:
    args = pickle.load(file)


def train_model(path, data):
    train_data = os.path.join(os.getenv("DATADIR", ""), path + data)
    model = fasttext.train_supervised(input=train_data, epoch=int(args['epochs']), lr=float(args['learning_rate']),
                                      dim=int(args['dimensions']), minCount=int(args['minimum_word_count']),
                                      wordNgrams=int(args['word_ngrams']), minn=int(args['min_char_grams']),
                                      maxn=int(args['max_char_grams'],
                                               saveOutput=True))
    model_parms = {"epochs": args['epochs'], "learning_rate": args['learning_rate'],
                   "dimensions": args['dimensions'], "minimum_word_count": args['minimum_word_count'],
                   "word_ngrams": args['word_ngrams'], "min_char_grams": args['min_char_grams'],
                   "max_char_grams": args['max_char_grams']}
    mlflow.start_run()
    mlflow.log_params(model_parms)

    if args['model_quantize'] == 'yes':
        model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
        print("saving model in progress")
        model.save_model(args['model_directory'] + args['model_name'] + ".bin")

        mlflow.log_model(path=model, name=args['model_name'])
        print("quantized model saved")
    else:
        print("saving model in progress")
        model.save_model(args['model_directory'] + args['model_name'] + ".bin")
        print("model saved")

# main train model step
def run_trainmodel():
    train_model(args['data_directory'], 'traindata_preprocessed.txt')
