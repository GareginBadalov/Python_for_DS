import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 51)
print(t)

f = np.cos(t)
print(f)
plt.plot(t,f,color="green")
plt.show()

plt.axis([0.5, 9.5, -2.5, 2.5])
plt.plot(t,f,color="green")
plt.title('График f(t)')
plt.xlabel('Значения t')
plt.ylabel('Значения f')
plt.show()