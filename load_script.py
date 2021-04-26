from predict import normalize_pixels
from predict import load_df
from pprint import pprint
import pandas as pd
import argparse
import sys
import numpy as np
import autokeras as ak
from tensorflow import keras
import joblib

# load model
loaded_model = keras.models.load_model("./autokeras_out_2/", custom_objects=ak.CUSTOM_OBJECTS)

# load validation_set
train_df = load_df('./training_aj1.csv')
df_mean = train_df['average_sale_price'].mean()

df = load_df('./validation_aj1.csv')
df = normalize_pixels(df)
df_x = df.drop(['average_sale_price','ticker'], axis=1)
df_y = df['average_sale_price']

df_y_hat = loaded_model.predict(df_x, verbose=0)
df_y_hat = [x[0] for x in df_y_hat]
print(df_y_hat)
print(list(df_y))
pprint(list(zip(df_y, df_y_hat)))

mae = keras.metrics.MeanAbsoluteError()

mae.update_state(df_y, df_y_hat)

print('using the model mae\t', mae.result().numpy())

df_mean_y_hat = [df_mean for x in df_y]
mae.update_state(df_y, df_mean_y_hat)

print('guessing the mean mae\t', mae.result().numpy())

print(df_mean)
