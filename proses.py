## FUNGSI PREPROCESSING
import pandas as pd
import string
import re
from keras.src.utils import pad_sequences
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def preprocessing(df):
    # Menjadikan huruf kecil
    df['Title'] = df['Title'].str.lower()

    # Fungsi untuk mengganti spasi dengan tanda baca
    def space_to_punct(text):
        for punct in string.punctuation:
            text = text.replace(punct, f' {punct} ')

        text = re.sub(' +', ' ', text)
        return text

    # Mengaplikasikan fungsi space_to_punct ke kolom 'Title'
    df['Title'] = df['Title'].apply(space_to_punct)

    # Fungsi untuk melakukan preprocessing
    def preprocess_text(text):
        # Menghapus angka
        text = re.sub(r"\d+", "", text)
        # Menghapus tanda baca
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Menghapus spasi berlebih
        text = text.strip()
        text = re.sub('\s+', ' ', text)
        return text

    # Mengaplikasikan preprocessing ke kolom 'Title'
    df['Title'] = df['Title'].apply(preprocess_text)

    # Download stopwords jika belum diunduh
    nltk.download('stopwords')

    # Mengambil stopwords dalam bahasa Indonesia
    list_stopwords = set(stopwords.words('indonesian'))

    # Menghapus stopwords dari setiap kalimat di kolom 'Title'
    df['Title'] = df['Title'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in list_stopwords]))

    return df

## FUNGSI WORDCLOUD
def generate_and_display_wordcloud(data_df, label_value):
    # Filter data berdasarkan label
    filtered_data = data_df[data_df['hoax'] == label_value]

    # Gabungkan semua teks dari kolom 'title' yang memiliki label tertentu menjadi satu teks panjang
    text = " ".join(filtered_data['Title'])

    # Membuat Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text)

    # Menampilkan Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

## FUNGSI TRAIN TEST SPLIT
def split_data(features, labels, test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    return x_train,x_test, y_train, y_test

## FUNGSI PLOT AKURASI DAN LOSS
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()

## FUNGSI EVALUATE DAN CONFUSSION MATRIX
def evaluate_and_visualize(model, x_train, y_train, X_test, y_test):
    # Print accuracy on training and testing data
    train_accuracy = model.evaluate(x_train, y_train)[1] * 100
    test_accuracy = model.evaluate(X_test, y_test)[1] * 100
    print("Accuracy of the model on Training Data is - ", train_accuracy)
    print("Accuracy of the model on Testing Data is - ", test_accuracy)

    # Predictions on test data
    predictions = model.predict(X_test)
    predictions_integer = (predictions > 0.5).astype(int)

    print("Sample Predictions:", predictions_integer[:5])

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, predictions_integer))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions_integer)
    cm = pd.DataFrame(cm, index=['Real', 'Hoax'], columns=['Real', 'Hoax'])

    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=['Real', 'Hoax'],
                yticklabels=['Real', 'Hoax'])
    plt.title('Confusion Matrix')
    plt.show()

## FUNGSI PREDICT TEST KALIMAT
PADDING = 'post'
def predict_sentences(sentences, model, tokenizer, max_len):
    # Tokenize and pad the input sentences
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding=PADDING, maxlen=max_len)

    # Make predictions using the model
    predictions = model.predict(padded)

    # Convert predictions to binary labels (0 or 1)
    binary_predictions = (predictions > 0.5).astype(int)

    return binary_predictions