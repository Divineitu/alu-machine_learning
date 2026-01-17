#!/usr/bin/env python3
'''
Plots y = x^3 as a red line graph
'''
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3

plt.plot(y, color="red")
plt.xlim(0, 10)
plt.show()
