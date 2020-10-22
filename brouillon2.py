import numpy as np
import matplotlib.pyplot as plt

a = [1,None,2,3, None,4]
for i in range(len(a)):
    if not a[i]:
        a[i] = a[i-1]

print(a)
