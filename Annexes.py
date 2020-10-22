#############################################
### Comparaison registre

import numpy as np
import matplotlib.pyplot as plt



######################################
# Calcul fréquence de battements
f0 = 130.0
d = 10.0
fc = f0*2**(d/(12.0*100))
print(2/(fc-f0))


######################################
# Longueur de la fenêtre d'analyse
import numpy as np

f = 77.0
sr = 22050
N = 1.5 * sr / (f * (2**(1.0/(12*8)) - 1))
print(N/sr)

#######################
# Transpositions

# DicTranspo = {'Original':[0.45,0.32,0.61,0.55], '+ Octave': [0.34,0.21,0.34,0.23], '- Octave':[0.71,0.62,0.73,0.67], '- 4te juste':[0.69,0.47,0.59,0.49],'- 3ce min':[0.55,0.29,0.44,0.64], '+ 5te juste':[0.45,0.26,0.35,0.35], '+ 4te juste':[0.37,0.26,0.37,0.28], '+ 3ce min':[0.49,0.27,0.33,0.30], '+ 2de Maj':[0.48,0.27,0.36,0.39], '- 2de Maj':[0.56,0.43,0.50,0.55], '- 5te juste':[0.76,0.44,0.50,0.67], '- 2de min':[0.55,0.40,0.51,0.57], '- 3ce Maj':[0.72,0.59,0.60,0.66], '- 5te dim':[0.68,0.41,0.51,0.70],'- 6te min':[0.64,0.47,0.52,0.56], '- 6te Maj':[0.80,0.62,0.50,0.65], '- 7e min':[0.68,0.63,0.63,0.64], '- 7e Maj':[0.75,0.59,0.76,0.76]}
# ListeTranspoU = ['+ 2de Maj','+ 3ce min', '+ 4te juste', '+ 5te juste', '+ Octave']
# ListeTranspoD = ['Original','- 2de min','- 2de Maj','- 3ce min','- 3ce Maj', '- 4te juste', '- 5te dim','- 5te juste', '- 6te min', '- 6te Maj', '- 7e min', '- 7e Maj','- Octave']
# ListeTranspoD.reverse()
# ListeTranspo = ListeTranspoD + ListeTranspoU
# labels = ListeTranspo
# # for int in ListeTranspoU:
# #     print(DicTranspo[int])
# #
# ench = []
# for i in range(4):
#     ench.append([DicTranspo[int][i] for int in labels])
#
# width = 0.35       # the width of the bars: can also be len(x) sequence
# bottom = [0 for i in range(len(labels))]
# fig, ax = plt.subplots()
# for i in range(3,4):
#     ax.bar(labels, ench[i],width,bottom = bottom,label='Ench {}'.format(i+1))
#     for int in range(len(labels)):
#         bottom[int]+=ench[i][int]
#
#
# ax.set_ylabel('DiffRoughness sum')
# ax.set_title('Crucis Transposition')
# ax.legend()
#
# plt.show()
######################################

# Représentation de la courbe fondamentale de dissonance
# Double exponentielle décroissante

# import matplotlib.pyplot as plt
# import numpy as np
# import params
#
# f1, f2, f3 = 130, 261, 523
# y = np.arange(0,1,0.001)
# x1, x2, x3 = f1*2**np.arange(0,1,0.001),f2*2**np.arange(0,1,0.001), f3*2**np.arange(0,1,0.001)
#
# def dissonance(f1,f2):
#     # diss = np.exp(-3.5*0.24*(f2-f1)/(0.021*f1+19))-np.exp(-5.75*0.24*(f2-f1)/(0.021*f1+19))
#     b1, b2 = 3.5, 4.0
#     if params.mod_rough == 'sethares + KK':
#         diss =  np.exp(-b1*(1/2.27)*(np.log(b2/b1)/(b2-b1))*(f2-f1)/(f1**(0.477)))-np.exp(-b2*(1/2.27)*(np.log(b2/b1)/(b2-b1))*(f2-f1)/(f1**(0.477)))
#     elif params.mod_rough == 'sethares':
#         diss =  np.exp(-b1*(f2-f1)*(0.24/(0.021*f1 + 19)))-np.exp(-b2*(f2-f1)*(0.24/(0.021*f1 + 19)))
#     return diss
#
# d1, d2, d3 = dissonance(f1,x1), dissonance(f2,x2), dissonance(f3,x3)
#
# plt.figure()
# plt.plot(12*y,d1,12*y,d2,12*y,d3)
# plt.legend(['%.0f' %int(f1) +' Hz (C3)', '%.0f' %int(f2) +' Hz (C4)', '%.0f' %int(f3) +' Hz (C5)'])
# # print(x[np.argmax(diss)])
# for i in range(13):
#     plt.vlines(i, 0, np.max(d1), linestyle='--')
# plt.xlabel('intervalle en demi-tons')
# plt.ylabel('Rugosité')
# plt.title('Courbe de rugosité entre sons simples')
# plt.show()


######################################
# # Gaussienne
# import matplotlib.pyplot as plt
# import numpy as np
#
# f1, f2, f3 = 10,440,10000
# x1, x2, x3 = f1*np.arange(1,2,0.001),f2*np.arange(1,2,0.001), f3*np.arange(1,2,0.001)
#
# x = np.arange(1,2,0.001)
#
# def dissonance(f1,f2):
#     # diss = np.exp(-3.5*0.24*(f2-f1)/(0.021*f1+19))-np.exp(-5.75*0.24*(f2-f1)/(0.021*f1+19))
#     σ=0.01
#     f0=440
#     diss =  np.exp(-(f2/f1 - (1 + 2.27*f1**(-0.523)))**2 / (2*(σ* ((1+2.27*f1**(-0.523))/(1+2.27*f0**(-0.523))) )**2))
#     #diss =  np.exp(-b1*(1/2.27)*(np.log(b2/b1)/(b2-b1))*(f2-f1)/(f1**(0.477)))-np.exp(-b2*(1/2.27)*(np.log(b2/b1)/(b2-b1))*((f2-f1)/(f1**(0.477)))**2)
#     return diss
#
# y1, y2, y3 = dissonance(f1,x1), dissonance(f2,x2), dissonance(f3,x3)
# print(x[np.argmax(y2)]-1)
# plt.figure()
# plt.plot(x,y1,x,y2,x,y3)
# plt.legend(['%.2f' %f1, '%.2f' %f2, '%.2f' %f3])
# # print(x[np.argmax(diss)])
# for i in range(13):
#     plt.vlines(1+i/12, 0, np.max(y1), linestyle='--')
# plt.show()



######################################
# Representations abstraites
# import matplotlib.pyplot as plt
# import numpy as np
# import pickle
# import params
#
# with open ('liste1', 'rb') as fp:
#     liste1 = pickle.load(fp)
# with open ('liste2', 'rb') as fp:
#     liste2 = pickle.load(fp)
# with open ('liste1v', 'rb') as fp:
#     liste1v = pickle.load(fp)
# with open ('liste2v', 'rb') as fp:
#     liste2v = pickle.load(fp)
#
#
# color = params.color_abstr
#
# plt.figure()
# ax = plt.subplot()
#
# plt.plot(liste1v, liste2v, 'r'+'--')
# plt.plot(liste1v, liste2v, 'r'+'o', label = 'Violon')
# for i in range(len(liste1)):
#     ax.annotate(' {}'.format(i+1), (liste1[i], liste2[i]), color='black')
#
# plt.plot(liste1, liste2, 'b'+'--')
# plt.plot(liste1, liste2, 'b'+'o',label = 'Piano')
# for i in range(len(liste1)):
#     ax.annotate(' {}'.format(i+1), (liste1v[i], liste2v[i]), color='black')
#
# d1, d2 = 'diffConcordance', 'crossConcordance'
# plt.xlabel(d1[0].upper() + d1[1:])
# plt.ylabel(d2[0].upper() + d2[1:])
# plt.title('Cadence ' + ' (' + d1[0].upper() + d1[1:] + ', ' + d2[0].upper() + d2[1:] + ')')
# plt.legend(frameon=True, framealpha=0.75)
# plt.show()
