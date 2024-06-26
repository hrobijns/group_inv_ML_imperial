
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import model_to_dot
from sklearn.model_selection import train_test_split
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
SHodge = np.array(SHodge)[:, 1:2]  

# Define the deep sets model
def get_deep_sets_model(input_shape):
    input_data = Input(shape=input_shape)
    x = Dense(16, activation='relu')(input_data)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    adder = Lambda(lambda x: tf.keras.backend.sum(x, axis=1, keepdims=True), output_shape=(1,))
    x = adder(x)
    output = Dense(1)(x)  

    model = Model(inputs=input_data, outputs=output)
    model.compile(optimizer=Adam(lr=1e-3, epsilon=1e-3), loss='mean_squared_error', metrics=['accuracy'])

    return model

def train_network(X_train, y_train, X_test, y_test):
    model = get_network()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(
        X_train, y_train,
        epochs=999999,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    return model, history

if __name__ == '__main__':
    model = get_deep_sets_model(input_shape=(5,))
    print(model.summary())
    X_train, X_test, y_train, y_test = train_test_split(Sweights, SHodge, test_size=0.5)
    print(f' {train_network(X_train, y_train, X_test, y_test)}')
