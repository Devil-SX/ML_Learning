import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0, 2*np.pi, 100)
x = np.sin(t)
y = 0.1*np.cos(0.1*t)

plt.plot(t, x)
plt.ylabel('sin(t)')
ax = plt.gca()
ax2 = ax.twinx()
ax2.plot(t, y, 'r.')
ax2.set_ylabel('cos(0.1t)')

plt.show()