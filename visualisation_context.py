import matplotlib.pyplot as plt
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
