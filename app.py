import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Carga los datos preprocesados
inputs = np.load("inputs.npy")
outputs = np.load("outputs.npy").reshape(-1, 1)  # Reshape a 2D array con la segunda dimensión igual a 1
word_to_index = np.load("word_to_index.npy", allow_pickle=True).item()

vocab_size = len(word_to_index)

# División de datos en conjuntos de entrenamiento y validación
split_idx = int(0.8 * len(inputs))
train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
train_outputs, val_outputs = outputs[:split_idx], outputs[split_idx:]


# Creación del modelo
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=inputs.shape[1]))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(vocab_size, activation="softmax"))

# Compilación del modelo
model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
model.summary()

# Entrenamiento del modelo
callbacks = [EarlyStopping(monitor="val_loss", patience=3, verbose=1)]
history = model.fit(train_inputs, train_outputs, validation_data=(val_inputs, val_outputs), epochs=20, batch_size=128, callbacks=callbacks)

# Guardar el modelo entrenado
model.save("trained_model.h5")
