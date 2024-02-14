from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
model.summary()

# model = load_model('my_model.h5')

PADDING = 'post'
OOV_TOKEN = "<OOV>"
max_features = 12000
maxlen = 30

tokenizer = Tokenizer(num_words=max_features,oov_token = OOV_TOKEN)

sentence = ["CAK NUN SEBUT JOKOWI SEPERTI FIR’AUN KARENA DISURUH"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, padding=PADDING,maxlen=maxlen)
print(model.predict(padded))