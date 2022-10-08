#Importing modules
import numpy as np

#Function for computing cost
def compute_cost(x_train, y_train, w, b):
  m = len(x_train)
  cost = 0
  for i in range(m):
    f_wb = w * x_train[i] + b
    cost += (f_wb - y_train[i]) ** 2
  return cost / (2 * m)

#Function for computing gradient
def compute_gradient(x_train, y_train, w, b):
  m = len(x_train)
  dj_dw = 0
  dj_db = 0
  for i in range(m):
    f_wb = w * x_train[i] + b
    dj_dw += (f_wb - y_train[i]) * x_train[i]
    dj_db += (f_wb - y_train[i])
  return dj_dw/m, dj_db/m

#Function for gradient descent
def gradient_descent(x_train, y_train, w_init, b_init, alpha, num_iters):
  w = w_init
  b = b_init
  for i in range(num_iters):
    dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)
    w -= alpha * dj_dw
    b -= alpha * dj_db
  return w, b

#Data sets
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

#Initial w and b
w_init = 0
b_init = 0

#Some values for gradient descents
alpha = 0.01
num_iters = 100000

print(gradient_descent(x_train, y_train, w_init, b_init, alpha, num_iters))