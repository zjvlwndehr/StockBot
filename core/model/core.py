from keras.layers import Activation, Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('E:/Development/AI/StockBot/core/model/data/samsung.csv')
print(data.head())

high = data['High'].values
low = data['Low'].values
mid = (high + low) / 2
print(mid[:5])

Checkpoint = ModelCheckpoint(
    'E:/Development/AI/StockBot/core/model/model/stock.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
)

######### create window #########

seq_len = 50
sequence_length = seq_len + 1

result = []
print(f'mid len:{len(mid)}\nseq_len:{seq_len}\nsequence_length:{sequence_length}\n:{len(mid) - sequence_length}')
for index in range(len(mid) - sequence_length):
    result.append(mid[index: index + sequence_length])

# std = mid[int(round(np.array(mid).shape[0] * 0.9))]
# std = mid[len(mid)-1]
std = mid[int(round(len(mid) * 0.9))]
print(f"stdidx:{int(round(len(mid) * 0.9))}\nstd:{std}")

######### normalize #########

normalized = []
for window in result:
    normalized_window = [((float(p) / float(window[0])) - 1) for p in window] # normalize: first value is 0 as standard value
    normalized.append(normalized_window)

result = np.array(normalized)
print(result.shape)
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

print(f"x_train.shape:{x_train.shape}\ny_train.shape:{y_train.shape}\nx_test.shape:{x_test.shape}\ny_test.shape:{y_test.shape}")

model = Sequential()

try:
    model = load_model('E:/Development/AI/StockBot/core/model/model/stock.h5')
    
except:
    ######### build model #########

    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    print(model.summary())

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

######### denormalize #########

# std = 50000
print(f'std:{std}')
pred = (1+pred)*std

# pred = pred * (high[row + seq_len] - low[row + seq_len]) + low[row + seq_len] 
print(x_test.shape)
print(pred.shape)

######### plot #########
# x axis: date
# y axis: price
plt.plot(pred, color='red', label='Prediction')
plt.plot((1+y_test)*std, color='blue', label='Ground Truth')
# plt.plot(y_test * (high[row + seq_len] - low[row + seq_len]) + low[row + seq_len], color='blue', label='Ground Truth')
plt.legend(loc='best')
plt.show()

