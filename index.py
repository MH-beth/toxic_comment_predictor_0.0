import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import TextVectorization, LSTM, Dropout, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential


df = pd.read_csv(
    os.path.join('jigsaw-toxic-comment-classification-challenge', 'train.csv', 'train.csv')
)

X = df['comment_text']
Y = df[df.columns[2:]].values
MAX_FEATURES = 200000 #number of words in the vocabs

#vectorize the words
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

#MCSHBAP
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, Y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)

train = dataset.take(int(len(dataset)*0.7))
validation = dataset.skip(int(len(dataset)*0.7)).take(int(len(dataset)*0.2))
test = dataset.skip(int(len(dataset)*0.9)).take(int(len(dataset)*0.1))

model = Sequential()
model.add(Embedding(MAX_FEATURES+1, 32))
model.add(Bidirectional(LSTM(32, activation = "tanh")))
model.add(Dense(128, activation = "relu"))
model.add(Dense(256, activation = "relu"))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(6, activation= "sigmoid"))
model.compile(loss = "BinaryCrossentropy", optimizer = "Adam")
model.summary()

history = model.fit(train, epochs=1, validation_data=validation)

while True :
    input_text = input('Write something : ')
    input_text_vectorized = vectorizer(input_text)
    res = model.predict(np.expand_dims(input_text_vectorized, 0))
    print(res)



