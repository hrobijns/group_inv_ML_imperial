def get_network():
    inp = tf.keras.layers.Input(shape=(5,))
    prep = tf.keras.layers.Reshape((5,))(inp)
    h1 = tf.keras.layers.Dense(16, activation='relu')(prep)
    h2 = tf.keras.layers.Dense(32, activation='relu')(h1)
    #adding a layer to the network which sums the inputs
    summed = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(h2)
    out = tf.keras.layers.Dense(1, activation='linear')(summed)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001)
    )
    return model
