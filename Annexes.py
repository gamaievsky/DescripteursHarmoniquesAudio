
## Représentation de la courbe fondamentale de dissonance
# Double exponentielle décroissante
import matplotlib.pyplot as plt
import numpy as np

f1, f2, f3 = 263,783,4700
y = np.arange(0,1,0.001)
x1, x2, x3 = f1*2**np.arange(0,1,0.001),f2*2**np.arange(0,1,0.001), f3*2**np.arange(0,1,0.001)

def dissonance(f1,f2):
    # diss = np.exp(-3.5*0.24*(f2-f1)/(0.021*f1+19))-np.exp(-5.75*0.24*(f2-f1)/(0.021*f1+19))
    b1, b2 = 3.5, 4.0
    diss =  np.exp(-b1*(1/2.27)*(np.log(b2/b1)/(b2-b1))*(f2-f1)/(f1**(0.477)))-np.exp(-b2*(1/2.27)*(np.log(b2/b1)/(b2-b1))*(f2-f1)/(f1**(0.477)))
    #diss =  np.exp(-b1*(1/2.27)*(np.log(b2/b1)/(b2-b1))*(f2-f1)/(f1**(0.477)))-np.exp(-b2*(1/2.27)*(np.log(b2/b1)/(b2-b1))*((f2-f1)/(f1**(0.477)))**2)
    return diss

d1, d2, d3 = dissonance(f1,x1), dissonance(f2,x2), dissonance(f3,x3)

plt.figure()
plt.plot(y,d1,y,d2,y,d3)
plt.legend(['%.2f' %f1, '%.2f' %f2, '%.2f' %f3])
# print(x[np.argmax(diss)])
for i in range(13):
    plt.vlines(i/12, 0, np.max(d1), linestyle='--')
plt.show()



# Gaussienne
import matplotlib.pyplot as plt
import numpy as np

f1, f2, f3 = 10,440,10000
x1, x2, x3 = f1*np.arange(1,2,0.001),f2*np.arange(1,2,0.001), f3*np.arange(1,2,0.001)

x = np.arange(1,2,0.001)

def dissonance(f1,f2):
    # diss = np.exp(-3.5*0.24*(f2-f1)/(0.021*f1+19))-np.exp(-5.75*0.24*(f2-f1)/(0.021*f1+19))
    σ=0.01
    f0=440
    diss =  np.exp(-(f2/f1 - (1 + 2.27*f1**(-0.523)))**2 / (2*(σ* ((1+2.27*f1**(-0.523))/(1+2.27*f0**(-0.523))) )**2))
    #diss =  np.exp(-b1*(1/2.27)*(np.log(b2/b1)/(b2-b1))*(f2-f1)/(f1**(0.477)))-np.exp(-b2*(1/2.27)*(np.log(b2/b1)/(b2-b1))*((f2-f1)/(f1**(0.477)))**2)
    return diss

y1, y2, y3 = dissonance(f1,x1), dissonance(f2,x2), dissonance(f3,x3)
print(x[np.argmax(y2)]-1)
plt.figure()
plt.plot(x,y1,x,y2,x,y3)
plt.legend(['%.2f' %f1, '%.2f' %f2, '%.2f' %f3])
# print(x[np.argmax(diss)])
for i in range(13):
    plt.vlines(1+i/12, 0, np.max(y1), linestyle='--')
plt.show()
