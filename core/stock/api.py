'''
stock bot api
'''

class deposit():
    def __init__(self, id) -> None:
        self.__id = id
        self.__cash = 0
        self.__stock = 0

    def show(self):
        print(f'show deposit data. ID: {self.__id}')
        print(f'cash: {self.__cash}')
        print(f'stock: {self.__stock}')

    def deposit(self, amount):
        self.__cash += amount
        self.show()
    
    def withdraw(self, amount):
        self.__cash -= amount
        self.show()

    def stock_buy(self, amount):
        self.__stock += amount
        price = self.now_samsung()
        self.__cash -= amount * price
        print(f'sell {amount} stock at {price}')
        self.show()
    
    def stock_sell(self, amount):
        self.__stock -= amount
        price = self.now_samsung()
        self.__cash += amount * price
        print(f'sell {amount} stock at {price}')
        self.show()
    
    def now_samsung(self)->int:
        from cfscrape import create_scraper as cs
        from datetime import datetime as dtm
        from bs4 import BeautifulSoup
        from parse import parse
        import pandas as pd
        url = 'https://finance.yahoo.com/quote/005930.KS/'
        r = cs().get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        concated = ''.join(soup.find('div', {'class': 'D(ib) Mend(20px)'}).text.split('.')[0].split(','))
        return int(str(concated))

class ai():
    def __init__(self) -> None:
        from cfscrape import create_scraper
        import pandas
        
        self.cs = create_scraper()
        self.pd = pandas
    ######### NA #########
    # def update_csv(self):
    #     r = self.cs.get("https://query1.finance.yahoo.com/v7/finance/download/005930.KS?period1=1508457600&period2=1666224000&interval=1d&events=history&includeAdjustedClose=true")
    #     with open(f'E:/Development/AI/StockBot/core/model/data/s.csv', 'wb') as f:
    #         f.write(r.content)
    #     csv = self.pd.read_csv(f'E:/Development/AI/StockBot/core/model/data/s.csv')
    #     print(csv.head())
    def round(self, num):
        if num/100 - int(num/100) > 0.5:
            return int(num/100) * 100 + 100
        else:
            return int(num/100) * 100

    def tomorrow_samsung(self)->int:
        from keras.models import Sequential, load_model
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import datetime
        import os
        from datetime import datetime as dtm
        
        csv = pd.read_csv(f'E:/Development/AI/StockBot/core/model/data/samsung.csv')
        print(csv.tail())
        High = csv['High'].values[::-1][0:50][::-1]
        Low = csv['Low'].values[::-1][0:50][::-1]
        Date = csv['Date'].values[::-1][0:50][::-1]
        print(f"Date[:5] L {Date[:5]}")
        
        Mid = (High + Low) / 2
        print(f"Mid[:5] L {Mid[:5]}")
        
        std = Mid[0]
        print(f"std L : {std}   Date : {Date[0]}")
        ######### normalize #########
        Mid = np.array([i/Mid[0]-1 for i in Mid])        
        print(f"Mid(N)[:5] L {Mid[:5]}")
        try:
            model = load_model(f'E:/Development/AI/StockBot/core/model/model/{dtm.now().strftime("%Y-%m-%d")}.h5')
            pred = model.predict(np.array(Mid).reshape(1, 50, 1))
            return self.round(int((pred[0][0]+1)*std))
        except:
            return None

    def predict(self):
        from keras.layers import Activation, Dense, Dropout, LSTM
        from keras.callbacks import ModelCheckpoint
        from keras.models import Sequential, load_model
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import datetime
        import os

        data = pd.read_csv('E:/Development/AI/StockBot/core/model/data/s.csv')
        high = data['High'].values
        low = data['Low'].values
        mid = (high + low) / 2

        Checkpoint = ModelCheckpoint(
            f'E:/Development/AI/StockBot/core/model/model/{datetime.datetime.now().strftime("%Y-%m-%d")}.h5',
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

        model = Sequential()

        try:
            model = load_model(f'E:/Development/AI/StockBot/core/model/model/{datetime.datetime.now().strftime("%Y-%m-%d")}.h5')
        except:
            ######### build LSTM model #########
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
                callbacks=[Checkpoint]
            )

        ######### predict #########

        print('x_test.shape: ', x_test.shape)
        print('x_test[0].shape: ', x_test[0].shape)
        print('x_test[0]: ', x_test[0])
        pred = model.predict(x_test)
        std = mid[-sequence_length]
        print('std: ', std)
        plt.plot((1+pred)*std, color='red', label='Prediction')
        # plt.plot(y_test, color='blue', label='Ground Truth')
        # plt.plot(y_test * (high[row + seq_len] - low[row + seq_len]) + low[row + seq_len], color='blue', label='Ground Truth')
        plt.legend(loc='best')
        plt.show()

        ######### save as image #########

        try:
            os.mkdir(f'E:/Development/AI/StockBot/image')
        except:
            pass
        finally:
            pass
        # plt.savefig(f'E:/Development/AI/StockBot/image/test.png')
        plt.savefig(f'E:/Development/AI/StockBot/image/{datetime.datetime.now().strftime("%Y-%m-%d")}.png')

class stock():
    def __init__(self) -> None:
        pass
    
    def now_samsung(self)->int:
        # from requests import get
        from cfscrape import create_scraper as cs
        from datetime import datetime as dtm
        from bs4 import BeautifulSoup
        from parse import parse
        import pandas as pd
        url = 'https://finance.yahoo.com/quote/005930.KS/'
        r = cs().get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        concated = ''.join(soup.find('div', {'class': 'D(ib) Mend(20px)'}).text.split('.')[0].split(','))
        csv = pd.read_csv('E:/Development/AI/StockBot/core/model/data/s.csv')
        df = pd.DataFrame({'Date': [dtm.now().strftime("%Y-%m-%d-%H:%M:%S")], 'Price': [concated]})
        # csv = csv.append(df, ignore_index=True)
        csv = pd.concat([csv,df])
        csv.to_csv('E:/Development/AI/StockBot/core/model/data/s.csv', index=False)
        # print("Now Samsung[{0}]? {1}".format(df.values[0][0], df.values[0][1]))
        return int(str(concated))
