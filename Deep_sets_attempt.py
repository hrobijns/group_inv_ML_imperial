
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

# Define the deep sets model
def get_deep_sets_model(input_shape):
    input_data = Input(shape=input_shape)
    x = Dense(16, activation='relu')(input_data)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    adder = Lambda(lambda x: tf.keras.backend.sum(x, axis=1), output_shape=(lambda shape: (shape[0], shape[2])))
    x = adder(x)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=input_data, outputs=output)
    model.compile(optimizer=Adam(lr=1e-3, epsilon=1e-3), loss='mean_squared_error', metrics=['accuracy'])

    return model

# Assuming Sweights.shape[1] represents the input shape
model = get_deep_sets_model(input_shape=(5,))

# Visualize the model architecture (optional)
SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(Sweights, SHodge, test_size=0.2, random_state=42)

# Train the model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=20, min_lr=0.000001)

model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_data=(X_val, y_val), callbacks=[reduce_lr])
