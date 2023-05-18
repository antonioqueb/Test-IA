import nltk
from collections import Counter
import numpy as np
import re
import unicodedata

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def preprocess_text(text):
    text = text.lower()
    text = text.replace('_', ' ')
    text = remove_accents(text)
    return text

def tokenize_text(text):
    nltk.download("punkt")
    return nltk.word_tokenize(text)

def create_mappings(tokens):
    word_freq = Counter(tokens)
    vocab = sorted(word_freq, key=word_freq.get, reverse=True)
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    return word_to_index, index_to_word

def text_to_numeric(tokens, word_to_index):
    return [word_to_index[word] for word in tokens]

def create_sequences(numeric_sequence, sequence_length=25):
    inputs = []
    outputs = []
    for i in range(len(numeric_sequence) - sequence_length):
        inputs.append(numeric_sequence[i:i+sequence_length])
        outputs.append(numeric_sequence[i+1:i+sequence_length+1])
    return np.array(inputs), np.array(outputs)

if __name__ == "__main__":
    file_path = "dataset.txt"
    
    # Carga y preprocesa el texto
    text = load_text(file_path)
    text = preprocess_text(text)
    
    # Tokeniza el texto
    tokens = tokenize_text(text)
    
    # Crea mapeos de palabras a índices y viceversa
    word_to_index, index_to_word = create_mappings(tokens)
    
    # Convierte el texto tokenizado en una secuencia numérica
    numeric_sequence = text_to_numeric(tokens, word_to_index)
    
    # Divide el texto en secuencias de entrada y salida
    inputs, outputs = create_sequences(numeric_sequence)

    # Guarda los datos preprocesados y los mapeos en archivos npy
    np.save("inputs.npy", inputs)
    np.save("outputs.npy", outputs)
    np.save("word_to_index.npy", word_to_index)
    np.save("index_to_word.npy", index_to_word)

    print("Datos preprocesados guardados.")
