import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import time
import datetime as dt
from pandas import read_csv
from matplotlib import pyplot
from pandas import DataFrame
from numpy import array
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import telebot

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
 """
 Frame a time series as a supervised learning dataset.
 Arguments:
 data: Sequence of observations as a list or NumPy array.
 n_in: Number of lag observations as input (X).
 n_out: Number of observations as output (y).
 dropnan: Boolean whether or not to drop rows with NaN values.
 Returns:
 Pandas DataFrame of series framed for supervised learning.
 """
 n_vars = 1 if type(data) is list else data.shape[1]
 df = DataFrame(data)
 cols, names = list(), list()
 # input sequence (t-n, ... t-1)
 for i in range(n_in, 0, -1):
   cols.append(df.shift(i))
 names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
 # forecast sequence (t, t+1, ... t+n)
 for i in range(0, n_out):
   cols.append(df.shift(-i))
 if i == 0:
   names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
 else:
   names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
 # put it all together
 agg = pd.concat(cols, axis=1)
 agg.columns = names
 print(agg)
 # drop rows with NaN values
 if dropnan:
   agg.dropna(inplace=True)
 return agg

def calculate(open_val, high_val, low_val, close_val, volume_val, market_cap_val, ma20_val, ma50_val, ma200_val, rsi_val,open_val1, high_val1, low_val1, close_val1, volume_val1, market_cap_val1, ma20_val1, ma50_val1, ma200_val1):
  data = pd.DataFrame(np.array([(open_val, high_val, low_val, close_val, volume_val, market_cap_val, ma20_val, ma50_val, ma200_val, rsi_val),
                              (open_val1, high_val1, low_val1, close_val1, volume_val1, market_cap_val1, ma20_val1, ma50_val1, ma200_val1, 0)]),columns=['Open','High','Low','Close','Volume','Market Cap','MA(20)','MA(50)','MA(200)','RSI(14)(SMA)'])
  valuesdata = series_to_supervised(data.values, n_in=1, n_out=1, dropnan=True).values
  #print(valuesdata)
  train_X, train_y = valuesdata[:, :-1], valuesdata[:, -1]
  #print(train_X)
  train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
  #print(train_X)
  new_output = model.predict(train_X)
  #print(new_output)
  return new_output[0,0]
  return 0;
BOT_TOKEN = ''
bot = telebot.TeleBot(BOT_TOKEN)
model = keras.models.load_model('bitcoinmodel.h5')

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    
    bot.reply_to(message, "benvenuto nel bot, inserisci i parametri")


@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    user_input = message.text.split(',')
    num = calculate(float(user_input[0]), float(user_input[1]), float(user_input[2]), float(user_input[3]), float(user_input[4]), float(user_input[5]), float(user_input[6]), float(user_input[7]), float(user_input[8]), float(user_input[9]), float(user_input[10]), float(user_input[11]), float(user_input[12]), float(user_input[13]), float(user_input[14]), float(user_input[15]), float(user_input[16]), float(user_input[17]), float(user_input[18]))
    #num = calculate(
#998.325,1031.39,996.702,1021.79,187143686.4,16343198030,878.4710952,802.4752353,678.0509801,87.1612344,1021.75,1044.08,1021.6,1044.08,196321881.9,16552062818,890.9709524,809.1809804,679.4337164,0);
    bot.reply_to(message, num)


bot.infinity_polling()

