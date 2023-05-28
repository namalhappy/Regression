import pandas as pd
import numpy as np
import regression
import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter



data_file = input("Data file name: ")  # User input datafile
datafile = data_file + ".csv" # Adding extention .csv

data_path = "C:/Users/Windows/Documents/Datasets/"
datafile = data_path+datafile


# Data preprocessing 
r = regression.reg(datafile)     # Create a Regression class

X_train, X_test, y_train, y_test =  r.preData() 

print(X_train)
## Hyper tune LSTM parameters
# lstm_params = r.hyt_lstm()


error = r.sensAnalysis()
np.savetxt("sensitivity_errors.csv", error, delimiter=",")
## Hyper tune gru parameters
gru_params = r.hyt_gru()
np.savetxt("gru_params.csv", gru_params, delimiter=",")

## Hyper tune rnn parameters
rnn_params = r.hyt_rnn()
np.savetxt("rnn_params.csv", rnn_params, delimiter=",")

# for name in ("LSTM","GRU","RNN"):
r.single_reg(r = "LSTM",  epochs = 10)

# for 
error = r.sensAnalysis(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)