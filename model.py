import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pythainlp.tokenize import word_tokenize

with open('data/language.txt', 'r', encoding='utf-8') as file:
    text = file.read()

tokens = word_tokenize(text)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(tokens)
sequences = tokenizer.texts_to_sequences([tokens])[0]

X = []
y = []

sequence_length = 5
for i in range(len(sequences) - sequence_length):
    X.append(sequences[i:i + sequence_length])
    y.append(sequences[i + sequence_length])

X = np.array(X)
y = np.array(y)

X = pad_sequences(X, maxlen=sequence_length, padding='post')

y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=sequence_length),
    LSTM(100),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

model.fit(X, y, epochs=10)

input_text = "วันนี้"
input_sequence = word_tokenize(input_text)
input_sequence = tokenizer.texts_to_sequences([input_sequence])[0]
input_sequence = pad_sequences([input_sequence], maxlen=sequence_length, padding='post')

predicted = model.predict(input_sequence)
predicted_word = tokenizer.index_word[np.argmax(predicted)]
print(predicted_word)