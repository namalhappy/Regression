import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, SimpleRNN
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
from numpy.polynomial.polynomial import polyfit
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# to plot results
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# to save the models
from tensorflow import keras
import tensorflow as tf

class reg: 
    def __init__(self, data_file_name, date_col = 1, trainP = 0.7, testP = 0.3, randomS = 0):
        self.dfn = data_file_name
        self.randomS = randomS      # random state for trainig and tesing split
        self.trainP = trainP        # train percentage
        self.testP = testP          # test percentage
        self.date_col = date_col    # True / False
        # selqf.age = age
    
    def pt(self):
        print("File name: " + self.dfn)
        
    
    def preData(self):
    
        # Data file location
        temp = pd.read_csv(self.dfn)        
        print(temp.columns)             # Debug
        
        if self.date_col:
            in_data = temp.iloc[:, 1:-1]     # Remove Date and Outputs from the input dataset
        else:
            in_data = temp.iloc[:, :-1]
            
        out_data = temp.iloc[:, -1:]     # Remove Date and Outputs from the input dataset
         
        print(in_data.columns)
        print(out_data.columns)
        
        # Data Scalling
        X_scaler = MinMaxScaler() 
        Y_scaler = MinMaxScaler()
        
        # data.columns
        X_data = X_scaler.fit_transform(in_data)
        Y_data = Y_scaler.fit_transform(out_data)
        
        # print(X_data)
        # print(Y_data)
        
        self.X_scaler = X_scaler
        self.Y_scaler = Y_scaler
        
        self.X_data = X_data
        self.Y_data = Y_data
        
        X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, random_state=0)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return self.X_train, self.X_test, self.y_train, self.y_test
        ##Debug
        # print(self.X_train)
        # print(self.X_test)
        # print(self.y_train)
        # print(self.y_test)
        
    def lstm(self, units = 50, optimizer = 'adam'):
        """# **LSTM**"""        
        # The LSTM architecture
        regressor = Sequential()
        # First LSTM layer with Dropout regularisation
        regressor.add(LSTM(units=units, return_sequences=True, input_shape=(self.X_train.shape[1],1)))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, return_sequences=True))
        # Second LSTM layer
        regressor.add(Dropout(0.2))
        # Third LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        # Fourth LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))
        # The output layer
        regressor.add(Dense(units=1))
        # Compiling the LSTM
        regressor.compile(optimizer = optimizer, loss='mean_squared_error', metrics= "accuracy")
              
        return regressor
        
    def single_reg(self, batch_size = 16, epochs = 50, optimizer = "rmsprop", r = "LSTM"):
    
        if r == "LSTM": 
            temp = self.lstm(optimizer = optimizer)
        elif r == "GRU":
            temp = self.gru(optimizer = optimizer)
        elif r == "RNN":
            temp = self.rnn(optimizer = optimizer)
            
        temp.fit(self.X_train,self.y_train,epochs=epochs,batch_size=batch_size)
        
        model_name = "Models/"+ r + "_" + str(epochs)
        
        # save model
        temp.save(model_name)
        
        # load model
        model = keras.models.load_model(model_name)

        predicted_test = model.predict(self.X_test)
        predicted_train = model.predict(self.X_train)
        
        predicted_test = self.Y_scaler.inverse_transform(predicted_test)
        predicted_train = self.Y_scaler.inverse_transform(predicted_train)
        
        y_test = self.Y_scaler.inverse_transform(self.y_test)
        y_train = self.Y_scaler.inverse_transform(self.y_train)
        
        
        rmse_test = np.sqrt(mean_squared_error(y_test, predicted_test))
        rmse_train = np.sqrt(mean_squared_error(y_train, predicted_train))
        print("RMSE train: %f" % (rmse_test))
        print("RMSE test: %f" % (rmse_train))        
        
        plot_title_train = "Wind Power Generation - Train using " + r
        plot_title_test = "Wind Power Generation - Test using " + r
        
        ## Plot results R
        self.plotR(y_train,predicted_train, plot_title_train )
        self.plotR(y_test,predicted_test, plot_title_test)
            
    def plotR(self, x, y, title):
        # plt.figure(figsize = (10,10))
        plt.scatter(x, y)                
        plt.xlabel("Predicted Wind Power / Month (MWh) ")
        plt.ylabel("Actual Wind Power / Month (MWh)")
        plt.grid()      
        
        x = x.flatten()
        y = y.flatten()
        # print(x)        
        r_squared = r2_score(x, y)
        r_squared = round(r_squared, 2)
        print(r_squared)
        title = title + " (R2 = "+ str(r_squared)+")"
        plt.title(title)
        
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x+b, color='red')
        save_title = "Figure/"+title+ ".jpg"
        
        plt.savefig ( save_title ,  format = "png" ,  dpi = 300 ) 
        
        plt.show()
        
        
        
    def hyt_lstm(self):   
        
        batch_size = [16, 32]
        epochs = [10, 20]
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer = optimizer)
        
        print(param_grid)
        
        model = KerasClassifier(build_fn=self.lstm)

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

        grid_result = grid.fit(self.X_train, self.y_train)

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        
        return param
                
    def gru(self, units = 50, optimizer = 'adam'):
        """# **GRU**"""        
        # The GRU architecture
        regressor = Sequential()
        # First GRU layer with Dropout regularisation
        regressor.add(GRU(units=units, return_sequences=True, input_shape=(self.X_train.shape[1],1)))
        regressor.add(Dropout(0.2))
        regressor.add(GRU(units=50, return_sequences=True))
        # Second GRU layer
        regressor.add(Dropout(0.2))
        # Third GRU layer
        regressor.add(GRU(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        # Fourth GRU layer
        regressor.add(GRU(units=50))
        regressor.add(Dropout(0.2))
        # The output layer
        regressor.add(Dense(units=1))
        # Compiling the GRU
        regressor.compile(optimizer = optimizer, loss='mean_squared_error', metrics= "accuracy")
              
        return regressor
        
    def hyt_gru(self):   
        
        batch_size = [16, 32]
        epochs = [10, 20]
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer = optimizer)
        
        print(param_grid)
        
        model = KerasClassifier(build_fn=self.lstm)

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

        grid_result = grid.fit(self.X_train, self.y_train)

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
            
    def rnn(self, units = 50, optimizer = 'adam'):
        """# **SimpleRNN**"""        
        # The SimpleRNN architecture
        regressor = Sequential()
        # First SimpleRNN layer with Dropout regularisation
        regressor.add(SimpleRNN(units=units, return_sequences=True, input_shape=(self.X_train.shape[1],1)))
        regressor.add(Dropout(0.2))
        regressor.add(SimpleRNN(units=50, return_sequences=True))
        # Second SimpleRNN layer
        regressor.add(Dropout(0.2))
        # Third SimpleRNN layer
        regressor.add(SimpleRNN(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        # Fourth SimpleRNN layer
        regressor.add(SimpleRNN(units=50))
        regressor.add(Dropout(0.2))
        # The output layer
        regressor.add(Dense(units=1))
        # Compiling the SimpleRNN
        regressor.compile(optimizer = optimizer, loss='mean_squared_error', metrics= "accuracy")
              
        return regressor
        
    def hyt_rnn(self):   
        
        batch_size = [16, 32]
        epochs = [10, 20]
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer = optimizer)
        
        print(param_grid)
        
        model = KerasClassifier(build_fn=self.lstm)

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

        grid_result = grid.fit(self.X_train, self.y_train)

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        
        
        
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
            
            
            