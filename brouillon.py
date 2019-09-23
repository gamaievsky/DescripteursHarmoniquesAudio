from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from operator import truediv

l = [9,2,1,4,5,6]
l.sort()
print(l)


x = np.arange(-2,2,0.01)
a = 0.6
y = np.exp(-(x/a)**2)

plt.figure()
plt.plot(x,y)
plt.show()






import librosa
import librosa.display







l.remove(4)

print(l)





print(max(l))



for i in range(3):
    res = np.zeros((1,3))
    A = np.zeros((1,3))
    for j in range(3):
        A[0,j] = i
    print(A[0,i])

for k in range(3):
    print(res[0,k])


title = 'Palestrina'
#Palestrina, Cadence4VMaj
y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/code/DescripteursHarmoniquesAudio/'+title+'.wav', duration = 6)

"""
    def Tension(self):
        self.tension = np.zeros(self.n_frames)
        for p1 in range(self.n_pistes-2):
            for p2 in range(a+1, self.n_pistes-1):
                for p3 in range(b+1, self.n_pistes):
                    for f1 in range(self.Nf):
                        for f2 in range(self.Nf):
                            for f3 in range(self.Nf):

                                self.tension = self.tension + (self.chromPistesSync[p1][i1,:] * self.chromPistesSync[p2][i2,:] * self.chromPistesSync[p3][i3,:]) * np.abs()


"""
