'''
stock bot api
'''

from joblib import PrintTime


class deposit():
    def __init__(self, id) -> None:
        import pandas as pd
        import os
        from datetime import datetime as dtm
        try:
            print(f'[INFO] Searching for USER=[{id}] data...')
            os.path.exists(f'./core/stock/data/{str(id)}/log{id}.csv')
            df = pd.read_csv(f'./core/stock/data/{str(id)}/log{id}.csv')
            self.__cash = df['Cash'][-1]
            self.__tmp = 0
            self.__count = df['Count'][-1]
            self.__stock = df['Stock'][-1]
            self.__count = df['Count'][-1] 
            print(f'[INFO] USER=[{id}] data found!')
        except:
            print('[INFO]: No data.')
            self.__id = id
            self.__cash = 0
            self.__stock = 'samsumg electronics' # default; because this is test version
            self.__count = 0
            try:
                print('[INFO]: Creating data folder.')
                os.mkdir(f'./core/stock/data/{str(id)}')
            except:
                pass
            try:
                print('[INFO]: Creating log file.')
                df = pd.DataFrame([{'Date':dtm.now().strftime("%Y-%m-%d-%H:%M:%S"),'Id':self.__id, 'Cash':self.__cash, 'Count':self.__count, 'Stock':self.__stock}])
                df.to_csv(f'./core/stock/data/{str(id)}/log{id}.csv', index=False, header=True)
            except:
                pass
    def decorator(self, func):
        def wrapper(*args, **kwargs):
            print(f'[INFO]: {func.__name__} is running...')
            func(*args, **kwargs)
            print(f'[INFO]: {func.__name__} is done.')
        return wrapper

    def logger(self):
        import pandas as pd
        from datetime import datetime as dtm
        csv = [{'Date':dtm.now().strftime("%Y-%m-%d-%H:%M:%S"),'Id':self.__id, 'Cash':self.__cash, 'Count':self.__count, 'Stock':self.__stock}]
        df = pd.DataFrame(csv)
        print(f'[INFO]: Logger trying to save your(USER=[{self.__id}]) data.')
        try:
            print(df)
            df.to_csv(f'./core/stock/data/{self.__id}/log{self.__id}.csv', mode='a', index=False,header=False)
            print(f'[INFO]: Logger saved your(USER=[{self.__id}]) data.')
        except:
            print(f'[ERROR]: Logger failed to save your(USER=[{self.__id}]) data.')
            print(f'[ERROR]: Your transaction is not saved.')
    
    @decorator
    def deposit(self, amount):
        self.__tmp = self.__cash
        self.__cash += amount
        self.logger()
    
    def withdraw(self, amount):
        self.__tmp = self.__cash
        self.__cash -= amount
        self.logger()

    def stock_buy(self, amount):
        self.__tmp = self.__cash
        self.__count += amount
        price = self.now_samsung()
        self.__tmp = self.__cash
        self.__cash -= amount * price
        print(f'sell {amount} stock at {price}')
        self.logger()
    
    def stock_sell(self, amount):
        self.__count -= amount
        price = self.now_samsung()
        self.__cash += amount * price
        print(f'sell {amount} stock at {price}')
        self.logger()
    
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
        pass        
    ######### NA #########
    def update_csv_samsung(self)->None:
        from cfscrape import create_scraper as cs
        import pandas as pd
        from datetime import datetime as dtm
        try:
            print('[INFO]: Updating csv.')
            # print(f"https://query1.finance.yahoo.com/v7/finance/download/005930.KS?period1=1508457600&period2={str(int(dtm.now(tz=None).timestamp()))}&interval=1d&events=history&includeAdjustedClose=true")
            r = cs().get(f"https://query1.finance.yahoo.com/v7/finance/download/005930.KS?period1=1508457600&period2={str(int(dtm.now(tz=None).timestamp()))}&interval=1d&events=history&includeAdjustedClose=true")
            with open(f'./core/model/data/samsung{dtm.now().strftime("%Y-%m-%d")}.csv', 'wb') as f:
                f.write(r.content)
            print("[SUCCESS]: CSV updated.")
        except:
            print("[ERROR]: CSV update failed. Unknown error.")

    def round(self, num :int)->int:
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
        try:    
            csv = pd.read_csv(f'./core/model/data/samsung{dtm.now().strftime("%Y-%m-%d")}.csv')
            High = csv['High'].values[::-1][0:50][::-1]
            Low = csv['Low'].values[::-1][0:50][::-1]
            Date = csv['Date'].values[::-1][0:50][::-1]
            
            Mid = (High + Low) / 2
            
            std = Mid[0]
            Mid = np.array([i/Mid[0]-1 for i in Mid])
            try:
                model = load_model(f'./core/model/model/samsung{dtm.now().strftime("%Y-%m-%d")}.h5')
                pred = model.predict(np.array(Mid).reshape(1, 50, 1))
                return self.round(int((pred[0][0]+1)*std))
            except:
                print("[ERROR]: Model not found. Update model.")
                try:
                    self.update_model_samsung()
                    model = load_model(f'E:/Development/AI/StockBot/core/model/model/samsung{dtm.now().strftime("%Y-%m-%d")}.h5')
                    print("[SUCCESS]: Model updated automatically.")
                    pred = model.predict(np.array(Mid).reshape(1, 50, 1))
                    return self.round(int((pred[0][0]+1)*std))
                except:
                    print("[ERROR]: Model update failed. Unknown error.")
                    return 0
        except:
            return 0

    def update_model_samsung_force(self)->None:
        from datetime import datetime as dtm
        import os
        os.remove(f'./core/model/model/samsung{dtm.now().strftime("%Y-%m-%d")}.h5')
        print("[SUCCESS]: Model removed.")
        self.update_model_samsung()
    
    def update_model_samsung(self)->None:
        self.update_csv_samsung()
        from keras.layers import Activation, Dense, Dropout, LSTM
        from keras.callbacks import ModelCheckpoint
        from keras.models import Sequential, load_model
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from datetime import datetime as dtm
        import os

        data = pd.read_csv(f'./core/model/data/samsung{dtm.now().strftime("%Y-%m-%d")}.csv')
        high = data['High'].values
        low = data['Low'].values
        mid = (high + low) / 2

        seq_len = 50
        sequence_length = seq_len + 1
        Checkpoint = ModelCheckpoint(
            f'./core/model/model/samsung{dtm.now().strftime("%Y-%m-%d")}.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
        )

        result = []
        for index in range(len(mid) - sequence_length):
            result.append(mid[index: index + sequence_length])        
        
        ######### normalize #########
        normalized = []
        for window in result:
            normalized_window = [((float(p) / float(window[0])) - 1) for p in window] # normalize: first value is 0 as standard value
            normalized.append(normalized_window)

        result = np.array(normalized)
        np.random.shuffle(result)

        x_train = result[:, :-1]
        y_train = result[:, -1]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()

        try:
            model = load_model(f'E:/Development/AI/StockBot/core/model/model/samsung{dtm.now().strftime("%Y-%m-%d")}.h5')
        except:
            model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(100, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='linear'))
            model.compile(loss='mse', optimizer='rmsprop')
            model.fit(x_train, y_train, validation_split=0.1, batch_size=10, epochs=20, callbacks=[Checkpoint])
            model.save(f'E:/Development/AI/StockBot/core/model/model/samsung{dtm.now().strftime("%Y-%m-%d")}.h5')
            print("[SUCCESS]: Model updated.")
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
        return int(str(concated))
