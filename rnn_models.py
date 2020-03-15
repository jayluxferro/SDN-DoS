from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten, GRU, Activation, SimpleRNN
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K

def simpleRNN(input_data_shape, num_features):
    model = Sequential()
    model.add(SimpleRNN(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(SimpleRNN(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model

def simpleRNN2(input_data_shape, num_features):
    model = Sequential()
    model.add(SimpleRNN(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(SimpleRNN(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model

def lstm(input_data_shape, num_features):
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model

def lstm2(input_data_shape, num_features):
    model = Sequential()
    model.add(LSTM(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(LSTM(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model

def gru(input_data_shape, num_features):
    model = Sequential()
    model.add(GRU(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(GRU(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model

def gru2(input_data_shape, num_features):
    model = Sequential()
    model.add(GRU(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(GRU(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model


def lstm_gru(input_data_shape, num_features):
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(GRU(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model

def lstm_gru2(input_data_shape, num_features):
    model = Sequential()
    model.add(LSTM(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(GRU(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model


def gru_lstm(input_data_shape, num_features):
    model = Sequential()
    model.add(GRU(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model

def gru_lstm2(input_data_shape, num_features):
    model = Sequential()
    model.add(GRU(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(LSTM(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model


def simpleRNN_lstm(input_data_shape, num_features):
    model = Sequential()
    model.add(SimpleRNN(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model

def simpleRNN_lstm2(input_data_shape, num_features):
    model = Sequential()
    model.add(SimpleRNN(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(LSTM(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model


def lstm_simpleRNN(input_data_shape, num_features):
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(SimpleRNN(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model

def lstm_simpleRNN2(input_data_shape, num_features):
    model = Sequential()
    model.add(LSTM(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(SimpleRNN(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model


def simpleRNN_gru(input_data_shape, num_features):
    model = Sequential()
    model.add(SimpleRNN(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(GRU(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model

def simpleRNN_gru2(input_data_shape, num_features):
    model = Sequential()
    model.add(SimpleRNN(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(GRU(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model


def gru_simpleRNN(input_data_shape, num_features):
    model = Sequential()
    model.add(GRU(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(SimpleRNN(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model

def gru_simpleRNN2(input_data_shape, num_features):
    model = Sequential()
    model.add(GRU(500, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, input_data_shape], return_sequences=True))
    model.add(SimpleRNN(400, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(num_features, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae','mse','accuracy'])
    model.summary()

    return model
