from keras.layers import Activation, Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
# from requests import get
import cfscrape

def update_csv():
    # r = get("https://query1.finance.yahoo.com/v7/finance/download/005930.KS?period1=1508457600&period2=1666224000&interval=1d&events=history&includeAdjustedClose=true")
    # r = get("https://query1.finance.yahoo.com/v7/finance/download/005930.KS")
    # r = cfscrape.create_scraper().get("https://query1.finance.yahoo.com/v7/finance/download/005930.KS")
    r = cfscrape.create_scraper().get("https://query1.finance.yahoo.com/v7/finance/download/005930.KS?period1=1508457600&period2=1666224000&interval=1d&events=history&includeAdjustedClose=true")
    print(r.content.decode('utf-8'))
    with open(f'E:/Development/AI/StockBot/core/model/data/s.csv', 'wb') as f:
        f.write(r.content)
    # csv = pd.read_csv(f'E:/Development/AI/StockBot/core/model/data/s.csv')
    # print(csv.values[0])
    # with open(f'E:/Development/AI/StockBot/core/model/data/samsung.csv', 'a') as f:
    #     # f.write(f'{csv.values[0][0]},{csv.values[0][1]},{csv.values[0][2]},{csv.values[0][3]},{csv.values[0][4]},{csv.values[0][5]},{csv.values[0][6]}
    #     f.write(' ')
    #     csv.to_csv(f, header=False, index=False)
    #     # f.write(csv.values[0][0] + ',' + csv.values[0][1] + ',' + csv.values[0][2] + ',' + csv.values[0][3] + ',' + csv.values[0][4] + ',' + csv.values[0][5] + ',' + csv.values[0][6])
    



def train():
    data = pd.read_csv('E:/Development/AI/StockBot/core/model/data/samsung.csv')
    high = data['High'].values
    low = data['Low'].values
    mid = (high + low) / 2

    Checkpoint = ModelCheckpoint(
        f'E:/Development/AI/StockBot/core/model/model/{datetime.datetime.now()}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
    )

    ######### create window #########

    seq_len = 50
    sequence_length = seq_len + 1

    result = []
    for index in range(len(mid) - sequence_length):
        result.append(mid[index: index + sequence_length])

    ######### normalize #########

    normalized = []
    for window in result:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window] # normalize: first value is 0 as standard value
        normalized.append(normalized_window)

    result = np.array(normalized)
    ######### split train/test #########

    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_test = result[row:, -1]

    ######### build model #########

    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    ######### train model #########

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=10,
        epochs=20,
        # callbacks=[Checkpoint]
    )


    ######### predict #########

    pred = model.predict(x_test)

update_csv()