title = 'SchubertRecord'
import numpy as np
import os
if os.path.exists('Onset_given_'+title+'.txt'):
    print('Le .txt existe')
    onsets = []
    with open('Onset_given_SchubertRecord.txt','r') as f:
        for line in f:
            l = line.split()
            onsets.append(float(l[0]))
    onset_frames = np.asarray(onsets)
    print(isinstance(onset_frames, np.ndarray))
#Avec ma méthode de visualisation
elif os.path.exists('Onset_given_'+title):
    print('Le fichier sans format existe')
else: print("Aucun fichier n'existe")





# # Représentation de la courbe fondamentale de dissonance
# Double exponentielle décroissante
# import matplotlib.pyplot as plt
# import numpy as np
#
# f1, f2, f3 = 27.5, 261, 2793
# y = np.arange(0,1,0.001)
# x1, x2, x3 = f1*2**np.arange(0,1,0.001),f2*2**np.arange(0,1,0.001), f3*2**np.arange(0,1,0.001)
#
# def dissonance(f1,f2):
#     # diss = np.exp(-3.5*0.24*(f2-f1)/(0.021*f1+19))-np.exp(-5.75*0.24*(f2-f1)/(0.021*f1+19))
#     b1, b2 = 3.5, 4.0
#     diss =  np.exp(-b1*(1/2.27)*(np.log(b2/b1)/(b2-b1))*(f2-f1)/(f1**(0.477)))-np.exp(-b2*(1/2.27)*(np.log(b2/b1)/(b2-b1))*(f2-f1)/(f1**(0.477)))
#     #diss =  np.exp(-b1*(1/2.27)*(np.log(b2/b1)/(b2-b1))*(f2-f1)/(f1**(0.477)))-np.exp(-b2*(1/2.27)*(np.log(b2/b1)/(b2-b1))*((f2-f1)/(f1**(0.477)))**2)
#     return diss
#
# d1, d2, d3 = dissonance(f1,x1), dissonance(f2,x2), dissonance(f3,x3)
#
# plt.figure()
# plt.plot(12*y,d1,12*y,d2,12*y,d3)
# plt.legend(['%.2f' %f1 +'Hz', '%.2f' %f2 +'Hz', '%.2f' %f3 +'Hz'])
# # print(x[np.argmax(diss)])
# for i in range(13):
#     plt.vlines(i, 0, np.max(d1), linestyle='--')
# plt.xlabel('intervalle en demi-tons')
# plt.ylabel('Rugosité')
# plt.show()


#
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
#



#
# # Representations abstraites
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
