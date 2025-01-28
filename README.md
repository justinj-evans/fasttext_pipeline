# Project Information
A text classification workflow using fasttext.

All files run through a central file called 'master.py' in the run_scripts folder. A local dictionary called 'args.txt' is used to define what and how to run steps in the workflow (ex. where find the data, number of epochs for building the model) . Using update_arguements in the main file will change these default parameters. 

Defining directories is important as most functions use a (path + name) format, to retrieve your files.

# Usage
Example: classifying the genre of wiki movie plots

[1] update_arguements() - We define what the file looks like (ie. what are our codes and text).

[2] run_preprocess() - We split the dataset into a train, valid, test split. Each split is preprocessed and put into the 'data' directory.

[3] update_arguements() - We define what our model will be called (ex. model_1.3_test2) and provide a description for that run.

[4] run_tunemodel() - We hyperparameter tune a set of fasttext parameters defined in 'tune_variables[]'. We also define what metric is being tuned for (accuracy, F1 score, precision, recall), and the best set of parameters is then loaded into the args dictionary.

[5] run_trainmodel() - We then train the model based on either the tuned parameters, our own 'update_arguement' parameters, or the default. The model is then saved in the 'model' directory.

[6] run_testmodel() - Using the model that we made we predict on the valid and test datasets and save these files to the 'data' directory.

[7] run_metrics() - Based on the tested datasets we generate an accuracy, weighted-F1, weighted-precision, weight-recall, and a bootstrapped accuracy score.

[8] setthreshold_testdata() - Finally, to mimic threshold selection we evaluate the valid dataset (select a threshold nearest 5% error), then apply this threshold to the test dataset; returning the autocoding and error rate.
