import matplotlib.pyplot as plt
import numpy as np
import librosa

seuil = 170 #Hz
N1, N2 = 'G3', 'G#3'
f1, f2 = librosa.note_to_hz(N1), librosa.note_to_hz(N2)
l1, l2 = [], []
moy, diff = [], []
for i in range(1,11):
    l1.append(i*f1)
    l2.append(i*f2)
for k1 in range(len(l1)):
    for k2 in range(len(l2)):
        d = np.abs(l1[k1]-l2[k2])
        if d<=seuil:
            moy.append((l1[k1]+l2[k2])/2)
            diff.append(d)

plt.figure()
plt.plot(moy, diff,'o')
plt.show()
