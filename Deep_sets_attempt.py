
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import model_to_dot
from IPython.display import SVG

# Load your data
Sweights, SHodge = [], []
with open('/content/Topological_Data.txt','r') as file:
    for idx, line in enumerate(file.readlines()[1:]):
        if idx % 6 == 0: 
            Sweights.append(eval(line))
        if idx % 6 == 2: 
            SHodge.append(eval(line))

Sweights = np.array(Sweights)
SHodge = np.array(SHodge)

def deep_sets_model(input_shape):
    input_vec = Input(shape=input_shape)
    x = Dense(16, activation='relu')(input_vec)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    latent_rep = Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)  # Sum over the latent space
    output_vec = Dense(1, activation='sigmoid')(latent_rep)  # Assuming binary classification
    model = Model(input_vec, output_vec)
    return model


def get_network():
    inp = tf.keras.layers.Input(shape=(5,))
    prep = tf.keras.layers.Reshape((5,))(inp)
    h1 = tf.keras.layers.Dense(16, activation='relu')(prep)
    h2 = tf.keras.layers.Dense(32, activation='relu')(h1)
    h3 = tf.keras.layers.Dense(16, activation='relu')(h2)
    out = tf.keras.layers.Dense(2, activation='linear')(h3)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model

def train_network(X_train, y_train, X_test, y_test):
    model = get_network()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(
        X_train, y_train,
        epochs=1000,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    return history


if __name__ == '__main__':
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(Sweights, SHodge, test_size=0.5)

    # Train and evaluate the model
    test_accuracy = train_network(X_train, y_train, X_test, y_test)
    print(f'Test Accuracy of Deep Sets-like model: {test_accuracy}')
