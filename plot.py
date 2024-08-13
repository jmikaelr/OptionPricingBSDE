import matplotlib.pyplot as plt
import numpy as np

# Data points
x = [8, 20, 150]
y1 = [7.9631,8.3299,8.7631]  # MC-LSM
y2 = [8.0358,8.3032,8.5549]   # DNN

# Reference value (considered as the exact or best-known value)
reference_value = 8.068

# Calculate absolute errors
err_y1 = np.abs(reference_value - np.array(y1))
err_y2 = np.abs(reference_value - np.array(y2))

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, err_y1, label='MC-LSM', color='blue', marker='o')
plt.plot(x, err_y2, label='DNN', color='red', marker='o', linestyle='--')

#plt.plot(x, y1, label='MC-LSM', color='blue', marker='o')
#plt.plot(x, y2, label='DNN', color='red', marker='o', linestyle='--')
# Logarithmic scale for y-axis to better show convergence
plt.yscale('log')

plt.xlabel('Number of Time Steps (N)')
plt.ylabel('Logarithmic scaled Absolute Error')
#plt.ylabel('Option Price')
plt.title('Convergence of American Max-Call Option, Varying Time Steps (N)')
#plt.title('Option Price vs. Number of Time Steps (N)')
plt.legend()
plt.grid(True)

plt.show()

