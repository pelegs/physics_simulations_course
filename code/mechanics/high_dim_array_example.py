import numpy as np

num_steps = 10
num_objects = 1
th = np.zeros((num_objects, num_steps))
th[:, 0] = np.random.uniform(-1, 1, size=num_objects)

# print(th)
for t in range(1, num_steps):
    th[:, t] = np.sin(th[:, t - 1])
print(th)
