# a simple PINN aimed at solving the heat equation, i.e. del(u) / del(t) = alpha * del^2(u) / del(x)^2
################################################################################
# import relevant libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

################################################################################
def generate_colation_points(x_lim_lower, x_lim_higher, t_lim_lower, t_lim_higher, number_points):
  # generate colocation points
  x_colocation = np.random.uniform(low=x_lim_lower, high=x_lim_higher, size=(number_points, 1)).astype(np.float32)
  t_colocation = np.random.uniform(low=t_lim_lower, high=t_lim_higher, size=(number_points, 1)).astype(np.float32)
  x_colocation_tf = tf.convert_to_tensor(x_colocation, dtype=tf.float32)
  t_colocation_tf = tf.convert_to_tensor(t_colocation, dtype=tf.float32)
  return x_colocation_tf, t_colocation_tf

def generate_boundary_points():
    # boundary points for u(x=0.5, t=0)=1
    x_boundary1 = np.full((1, 1), 0.5).astype(np.float32)
    t_boundary1 = np.zeros((1, 1)).astype(np.float32)
    u_boundary1 = np.full((1, 1), 2).astype(np.float32)

    # boundary points for u(x=1, t=0)=0
    x_boundary2 = np.full((1, 1), 1.0).astype(np.float32)
    t_boundary2 = np.zeros((1, 1)).astype(np.float32)
    u_boundary2 = np.full((1, 1), 0.0).astype(np.float32)

    # concatenate the boundary points
    x_boundary = np.concatenate([x_boundary1, x_boundary2], axis=0)
    t_boundary = np.concatenate([t_boundary1, t_boundary2], axis=0)
    u_boundary = np.concatenate([u_boundary1, u_boundary2], axis=0)

    # convert to tensors
    x_boundary_tf = tf.convert_to_tensor(x_boundary, dtype=tf.float32)
    t_boundary_tf = tf.convert_to_tensor(t_boundary, dtype=tf.float32)
    u_boundary_tf = tf.convert_to_tensor(u_boundary, dtype=tf.float32)

    return x_boundary_tf, t_boundary_tf, u_boundary_tf

################################################################################
# define PINN
class PINN(tf.keras.Model):
  def __init__(self):
    super().__init__() # to access features of the parent class, tf.keras.Model     
    self.dense1 = tf.keras.layers.Dense(16, activation='tanh')
    self.dense2 = tf.keras.layers.Dense(32, activation='tanh')
    self.dense3 = tf.keras.layers.Dense(16, activation='tanh')
    self.output_layer = tf.keras.layers.Dense(1, activation=None)
  
  def call(self, inputs): # passes the inputs through the architecture defined above
    x = inputs[:, 0:1]
    t = inputs[:, 1:2]
    concat_input = tf.concat([x, t], axis=1)
    hidden1 = self.dense1(concat_input)
    hidden2 = self.dense2(hidden1)
    hidden3 = self.dense3(hidden2)
    output = self.output_layer(hidden3)
    return output

  def compute_gradients(self, x, t): #computes required gradients for minimising loss
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      tape.watch(t)
      u_pred = self(tf.concat([x, t], axis=1))
      u_x = tape.gradient(u_pred, x)
      u_xx = tape.gradient(u_x, x)
      u_t = tape.gradient(u_pred, t)
    del tape
    return u_xx, u_t                  

# define loss function
def loss(model, x_colocation, t_colocation, x_boundary, t_boundary, u_boundary):
    # compute loss at colocation points
    alpha = 0.01
    u_xx, u_t = model.compute_gradients(x_colocation, t_colocation)
    pde_loss = u_t - alpha * u_xx
    physics_loss = tf.reduce_mean(tf.square(pde_loss))

    # compute loss at boundary points
    u_boundary_pred = model(tf.concat([x_boundary, t_boundary], axis=1))
    boundary_loss = tf.reduce_mean(tf.square(u_boundary_pred - u_boundary))

    return physics_loss + boundary_loss # scale if necessary

################################################################################
# define training of the model
def train_network(x_colocation_tf, t_colocation_tf, x_boundary_tf, t_boundary_tf, u_boundary_tf, model=PINN(), num_epochs=501):
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  
  for epoch in range(num_epochs):
      with tf.GradientTape() as tape:
          loss_value = loss(model, x_colocation_tf, t_colocation_tf, x_boundary_tf, t_boundary_tf, u_boundary_tf)
  
      gradients = tape.gradient(loss_value, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
      if epoch % 100 == 0:
          print(f"Epoch {epoch}/{num_epochs-1}, Loss: {loss_value.numpy()}")
  return model

################################################################################
# define animation function which plots graph of u(x) over time - just for visualisation purposes.
def plot_PINN_prediction(model, x_lim_lower, x_lim_higher, t_lim_lower, t_lim_higher):
  
  x_test = np.linspace(x_lim_lower, x_lim_higher, 500).reshape(-1, 1)
  # create a figure and axis for the animation
  fig, ax = plt.subplots()
  line, = ax.plot(x_test, np.zeros_like(x_test))
  ax.set_xlabel('x')
  ax.set_ylabel('u')
  ax.set_title('PINN Prediction of $u(x,t)$')
  
  def animate(t):
      t_test = np.full_like(x_test, t)
      u_pred = model(tf.concat([x_test, t_test], axis=1)).numpy()
      line.set_ydata(u_pred)
      ax.set_title(f'PINN approximation of $u(x)$ at t={t:.2f}')
      return line,

  #to-do change 
  ax.set_xlim(x_lim_lower, x_lim_higher)
  ax.set_ylim(0, 5)
  
  # create the animation
  t_values = np.linspace(t_lim_lower, t_lim_higher, 400)
  ani = animation.FuncAnimation(fig, animate, frames=t_values, interval=20, blit=True)
  # save the animation as an MP4 file
  ani.save('pinn_prediction_animation.mp4', writer='ffmpeg')
  # display the animation
  plt.show()

################################################################################
# running the program:
if __name__ == '__main__':
  x_colocation_tf, t_colocation_tf = generate_colation_points(x_lim_lower=0, x_lim_higher=1, t_lim_lower=0, t_lim_higher=30, number_points=1000)
  x_boundary_tf, t_boundary_tf, u_boundary_tf = generate_boundary_points()
  model = train_network(x_colocation_tf, t_colocation_tf, x_boundary_tf, t_boundary_tf, u_boundary_tf)

  plot_PINN_prediction(model, x_lim_lower=0, x_lim_higher=1, t_lim_lower=0, t_lim_higher=50)
