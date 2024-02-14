import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('model_tanpa_stem.h5')

# Show the model architecture
model.summary()

# Function to preprocess input text and make prediction
def predict():
    input_text = entry.get()
    #label.configure(text=input_text)
    if not input_text:
        messagebox.showinfo("Info", "Please enter a sentence.")
        return

    PADDING = 'post'
    OOV_TOKEN = "<OOV>"
    maxlen = 30

    tokenizer = Tokenizer(num_words=15000, oov_token=OOV_TOKEN)
    #sentence = ["CAK NUN SEBUT JOKOWI SEPERTI FIR’AUN KARENA DISURUH"]
    tokenizer.fit_on_texts(input_text)
    sequences = tokenizer.texts_to_sequences([input_text])

    padded = pad_sequences(sequences, padding=PADDING, maxlen=maxlen)

    # Make prediction
    prediction = model.predict(padded)

    # Mengekstrak nilai dari array
    #prediction = float(prediction[0, 0])
    # Sekarang, prediction_value dapat digunakan sebagai variabel float biasa
    print(prediction)

    #result = ["hoax" if predictions > 0.5 else "not hoax"]
    if prediction > 0.5:
        prediction = "Hoax"
    else:
        prediction = "Real"
    #result = "Hoax" if prediction > 0.5 else "Not Hoax"
    messagebox.showinfo("Prediction Result", f"The sentence is likely: {prediction}")

# Create the main application window
app = tk.Tk()
app.title("Hoax Detector")

# Create GUI components
label = tk.Label(app, text="Judul Kalimat:")
label.pack()

entry = tk.Entry(app, width=50)
entry.pack()

predict_button = tk.Button(app, text="Prediksi", command=predict)
predict_button.pack()

# Start the Tkinter event loop
app.mainloop()


