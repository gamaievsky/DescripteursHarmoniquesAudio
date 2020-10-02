import os

import matplotlib.pyplot as plt
import pickle
import numpy as np

mem = 4
temps = range(12)
coeff = []
l_stab = [1.0/(i+1) for i in range(mem)]
s = sum(l_stab)
l_stab = [el/s for el in l_stab]

for t in temps:
    if t<mem:
        l = [1.0/(i+1) for i in range(t+1)]
        s = sum(l)
        l = [el/s for el in l]
        coeff.append(l)
    else: coeff.append(l_stab)

plt.figure()
for t in temps[:-7]:
    plt.plot([t,t+1,t+1,t+2,t+2,t+3,t+3,t+4], [coeff[t][0],coeff[t][0],coeff[t+1][1],coeff[t+1][1],coeff[t+2][2],coeff[t+2][2],coeff[t+3][3],coeff[t+3][3]])
plt.plot([5,6,6,7,7,8], [coeff[5][0],coeff[5][0],coeff[5+1][1],coeff[5+1][1],coeff[5+2][2],coeff[5+2][2]])
plt.plot([6,7,7,8], [coeff[6][0],coeff[6][0],coeff[7][1],coeff[7][1]])
plt.plot([7,8], [coeff[7][0],coeff[7][0]])
plt.vlines(temps[:-3],0,1.1, color='k', alpha=0.9, linestyle='--')
plt.xlabel('Events')
plt.ylabel('Activation coefficient')
plt.show()








# from p5 import *
#
# def setup():
#     size(640, 360)
#     no_stroke()
#     background(204)
#
# def draw():
#     if mouse_is_pressed:
#         fill(random_uniform(255), random_uniform(127), random_uniform(51), 127)
#     else:
#         fill(255, 15)
#
#     circle_size = random_uniform(low=10, high=80)
#
#     circle((mouse_x, mouse_y), circle_size)
#
# def key_pressed(event):
#     background(204)
#
# run()



# def cyclic_perm(a):
#     n = len(a)
#     b = [[a[i - j] for i in range(n)] for j in range(n)]
#     return b
#
#
#
#
#
# l = cyclic_perm(A)
# print(l)
# l.sort(reverse = False)
# print(l)
#
# def prime_form(a):
#     def cyclic_perm(a):
#         n = len(a)
#         b = [[a[i - j] for i in range(n)] for j in range(n)]
#         return b
#     l = cyclic_perm(A)
#     l.sort(reverse = False)
#     return l[0]
#


# a = [4,3,5]
# ret= map(lambda x: 12-x, [4,3,5])
# print(ret)
# print(a)


# ret= map(lambda a:  map(lambda x: 12-x, a), l)
# print(ret)
#
# sorted(range(len(s)), key=lambda k: s[k])
