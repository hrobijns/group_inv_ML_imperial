# a simple PINN aimed at solving del^2(u)/del(x)^2 = sin(x)
################################################################################
# import relevant libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# training data generation: colocation points (at which physics loss is calculated) 

def generate_colocation_points(x_lim_lower, x_lim_higher, number_points):
  x_colocation = np.random.uniform(low=x_lim_lower, high=x_lim_higher, size=(number_points, 1)).astype(np.float32)
  x_colocation_tf = tf.convert_to_tensor(x_colocation, dtype=tf.float32)
  return x_colocation_tf

################################################################################
# define pre-processing layer - here we enforce periodicity, essentially forcing it to have solution (sin(x))

class CosSinLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CosSinLayer, self).__init__()

    def call(self, inputs):
        x = inputs[:, 0]  # assuming inputs is a batch of single values
        cos_x = tf.math.cos(x)
        sin_x = tf.math.sin(x)
        return tf.stack([cos_x, sin_x], axis=1)

################################################################################
# define PINN

class PINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.preprocess = CosSinLayer()
        self.dense1 = tf.keras.layers.Dense(16, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(32, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(16, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(1, activation=None)
  
    def call(self, inputs):
        preprocessed_inputs = self.preprocess(inputs)
        hidden1 = self.dense1(preprocessed_inputs)
        hidden2 = self.dense2(hidden1)
        hidden3 = self.dense3(hidden2)
        output = self.output_layer(hidden3)
        return output

    def compute_gradients(self, x): # computes required gradients for minimising loss
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            u_pred = self(x)
            u_x = tape.gradient(u_pred, x)
            u_xx = tape.gradient(u_x, x)
        del tape
        return u_xx

# define loss function
def loss(model, x_colocation):
    # compute loss at colocation points
    u_xx = model.compute_gradients(x_colocation)
    pde_loss = u_xx - tf.sin(x_colocation)
    physics_loss = tf.reduce_mean(tf.square(pde_loss))

    return physics_loss # scale if necessary

################################################################################
# define training of the model
def train_network(x_colocation_tf, model=PINN(), num_epochs=501):
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  
  for epoch in range(num_epochs):
      with tf.GradientTape() as tape:
          loss_value = loss(model, x_colocation_tf)
  
      gradients = tape.gradient(loss_value, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
      if epoch % 100 == 0:
          print(f"Epoch {epoch}/{num_epochs-1}, Loss: {loss_value.numpy()}")
  return model

################################################################################
# visualisation of the solution
def analytical_solution(x):
  return - np.sin(x) #we will still be out by a constant, but this can be enforced using b.c.s as in simple_PINN.py

def plot_PINN_prediction(x_lim_lower, x_lim_higher, model):
  x_test = np.linspace(x_lim_lower, x_lim_higher, 300).reshape(-1, 1).astype(np.float32)
  x_test_tf = tf.convert_to_tensor(x_test, dtype=tf.float32)
  u_pred = model(x_test_tf)
  plt.plot(x_test, analytical_solution(x_test), linestyle = ":", label='analytical')
  plt.plot(x_test, u_pred.numpy(), alpha=0.5, label='PINN predicted solution')
  plt.xlim(x_lim_lower,x_lim_higher)
  plt.xlabel('x')
  plt.ylabel('u')
  plt.legend()
  plt.show()

################################################################################
# running the program:
if __name__ == '__main__':
  x_colocation_tf = generate_colocation_points(0, 2 * np.pi, 1000)
  trained_model = train_network(x_colocation_tf, PINN())
  plot_PINN_prediction(0, 5, trained_model)
