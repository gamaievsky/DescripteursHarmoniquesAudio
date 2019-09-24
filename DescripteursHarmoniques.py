from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter, truediv
import statistics as stat
from scipy import signal
import math
#from scipy import signal

import librosa
import librosa.display

import params

WINDOW = params.WINDOW
BINS_PER_OCTAVE = params.BINS_PER_OCTAVE
STEP = 512
α = params.α
β = params.β
H = params.H
T = params.T
T_att = params.T_att

δ = params.δ #tension
S0 = params.S0 #dissonance
S1 = params.S1 #dissonance
S2 = params.S2 #dissonance
B1 = params.B1 #dissonance
B2 = params.B2 #dissonance

norm_spectre = params.norm_spectre
norm_conc = params.norm_conc
norm_concTot = params.norm_concTot

plot_onsets = params.plot_onsets
plot_pistes = params.plot_pistes
plot_chromDescr = params.plot_chromDescr
plot_descr = params.plot_descr

class SignalSepare:
    """ Prend en entrée en signal et le signal des pistes audio séparées. """

    def __init__(self, signal, sr, pistes, Notemin  = 'D3', Notemax = 'D9', delOnsets = [], addOnsets = []):
        self.y = signal
        self.pistes = pistes
        self.sr = sr
        self.n_pistes = len(pistes)
        self.Notemin = Notemin
        self.Notemax = Notemax
        self.delOnsets = delOnsets
        self.addOnsets = addOnsets
        self.n_bins = 0
        self.N = 0
        self.fmin = 0
        self.fmax = 0
        self.n_bins = 0
        self.n_att = 0
        self.n_frames = 0
        self.times = []
        self.onset_times = []
        self.onset_frames = []
        self.Chrom = []
        self.ChromDB = []
        self.chromSync = []
        self.chromSyncDB = []
        self.chromPistesSync = []
        self.energy = []
        self.chromConc = []
        self.concordance = []
        self.chromConcTot = []
        self.concordanceTot = []
        self.tension = []
        self.dissonance = []
        self.tensionSignal = []
        self.dissonanceSignal = []

    def DetectionOnsets(self):
        self.fmin = librosa.note_to_hz(self.Notemin)
        self.fmax = librosa.note_to_hz(self.Notemax)
        #Nmin = int((sr/(fmax*(2**(1/BINS_PER_OCTAVE)-1))))
        #Nmax = int((sr/(fmin*(2**(1/BINS_PER_OCTAVE)-1))))
        self.n_bins = int((librosa.note_to_midi(self.Notemax) - librosa.note_to_midi(self.Notemin))*BINS_PER_OCTAVE/12)
        self.Chrom = np.abs(librosa.cqt(y=self.y, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins, window=WINDOW))
        self.ChromDB = librosa.amplitude_to_db(self.Chrom, ref=np.max)
        self.N = len(self.Chrom[0])
        Diff = np.zeros((self.n_bins,self.N))
        Dev = np.zeros(self.N)
        for j in range(1,self.N):
            for i in range(self.n_bins):
                Diff[i,j] = np.abs(self.ChromDB[i,j]-self.ChromDB[i,j-1])
                Dev[j] = sum(Diff[:,j])

        # FONCTION DE SEUIL
        # Ajout de zéros en queue et en tête
        l = []
        Seuil = []
        Onsets = []
        for k  in range(int(H/2)):
            l.append(0)
        for val in Dev:
            l.append(val)
        for k  in range(int(H/2)):
            l.append(0)
        #Calcul de la médiane
        for i in range(self.N):
            Seuil.append(α + β*stat.median(l[i:i+H]))
            if Dev[i] > Seuil[i]:
                Onsets.append(i)


        self.times = librosa.frames_to_time(np.arange(self.N), sr=sr, hop_length=STEP)

        # FONCTION DE TRI SUR LES  ONSETS
        # Onsets espacés d'au moins T
        i=0
        while i<(len(Onsets)-1):
            while (i<(len(Onsets)-1)) and (self.times[Onsets[i+1]]< self.times[Onsets[i]]+T):
                if (Dev[Onsets[i+1]]-Seuil[Onsets[i+1]]) < (Dev[Onsets[i]]-Seuil[Onsets[i]]): del Onsets[i+1]
                #if (Dev[Onsets[i+1]]) < (Dev[Onsets[i]]): del Onsets[i+1]
                else: del Onsets[i]
            i=i+1

        # Suppression manuelle des onsets en trop (cela nécessite d'avoir affiché les onsets jusqu'ici détectés)
        self.delOnsets.sort(reverse = True)
        for o in self.delOnsets:
            Onsets.pop(o-1)

        # Ajout manuel des onsets
        for t in self.addOnsets:
            Onsets.append(librosa.time_to_frames(t, sr=sr, hop_length=STEP))
            Onsets.sort()






        self.onset_frames = librosa.util.fix_frames(Onsets, x_min=0, x_max=self.Chrom.shape[1]-1)
        self.onset_times = librosa.frames_to_time(self.onset_frames, sr=sr, hop_length = STEP)
        self.n_frames = len(self.onset_frames)-1


        #Synchronisation sur les onsets, en enlevant le début et la fin des longues frames
        self.chromSync = np.zeros((self.n_bins,self.n_frames))
        self.n_att = int(librosa.time_to_frames(T_att, sr=self.sr, hop_length = STEP))
        for j in range(self.n_frames):
            for i in range(self.n_bins):
                self.chromSync[i,j] = np.mean(self.Chrom[i][(self.onset_frames[j]+self.n_att):(self.onset_frames[j+1]-self.n_att)])
                self.chromSyncDB = librosa.amplitude_to_db(self.chromSync, ref=np.max)

        #Normalisation du spectre
        if norm_spectre:
            for j in range(self.n_frames):
                self.chromSync[:,j] = librosa.power_to_db(librosa.db_to_power(self.chromSync[:,j]) / np.sum(librosa.db_to_power(self.chromSync[:,j])))

        #Calcul de l'énergie
        for j in range(self.n_frames):
            self.energy.append(np.sum(np.multiply(self.chromSync[:,j], self.chromSync[:,j])))

        #Affichage
        if plot_onsets:
            plt.figure(figsize=(13, 7))
            ax1 = plt.subplot(3, 1, 1)
            librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times)
            plt.title('CQT spectrogram')

            plt.subplot(3, 1, 2, sharex=ax1)
            plt.plot(self.times, Dev, label='Deviation')
            plt.plot(self.times, Seuil, color='g', label='Seuil')
            plt.vlines(self.times[Onsets], 0, Dev.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
            plt.axis('tight')
            plt.legend(frameon=True, framealpha=0.75)

            ax1 = plt.subplot(3, 1, 3, sharex=ax1)
            librosa.display.specshow(self.chromSyncDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times)
            plt.show()


    def Clustering(self):
        """ Découpe et synchronise les pistes séparées sur les ONSET, stoque le spectrogramme
        synchronisé en construisant self.chromPistesSync"""

        #  Construction de chromPistesSync
        ChromPistes = []
        for k, voice in enumerate(self.pistes):
            ChromPistes.append(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)))
            self.chromPistesSync.append(np.zeros((self.n_bins,self.n_frames)))
            for j in range(self.n_frames):
                for i in range(self.n_bins):
                    self.chromPistesSync[k][i,j] = np.mean(ChromPistes[k][i][(self.onset_frames[j]+self.n_att):(self.onset_frames[j+1]-self.n_att)])
        #  Plot
        if plot_pistes:
            plt.figure(figsize=(13, 7))
            ax1 = plt.subplot(self.n_pistes,1,1)
            librosa.display.specshow(librosa.amplitude_to_db(self.chromPistesSync[0], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times)
            for k in range(1, self.n_pistes):
                plt.subplot(self.n_pistes, 1, k+1, sharex=ax1)
                librosa.display.specshow(librosa.amplitude_to_db(self.chromPistesSync[k], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times)
            plt.show()

    def Concordance(self):
        """Multiplie les spectres (cqt) des différentes pistes pour créer le spectre de concordance,
        et calcule la concordance en sommant sur les fréquences"""

        self.chromConc = np.zeros((self.n_bins,self.n_frames))
        for k in range(self.n_pistes-1):
            for l in range(k+1, self.n_pistes):
                self.chromConc = self.chromConc + np.multiply(self.chromPistesSync[k], self.chromPistesSync[l])
        self.concordance = self.chromConc.sum(axis=0)

        #Normalisation
        self.concordance = [x/y for x, y in zip(self.concordance, self.energy)]
        self.concordance[0]=0
        self.concordance[self.n_frames-1]=0
        #self.concordance[self.n_pistes-1]=0

    def ConcordanceTot(self):
        """Multiplie les spectres (cqt) des différentes pistes pour créer le spectre de concordance,
        et calcule la concordance en sommant sur les fréquences"""

        self.chromConcTot = np.ones((self.n_bins,self.n_frames))
        for k in range(self.n_pistes):
            self.chromConcTot = np.multiply(self.chromConcTot, self.chromPistesSync[k])
        self.concordanceTot = self.chromConcTot.sum(axis=0)

        # Normalisation...
        self.concordanceTot[0]=0
        self.concordanceTot[self.n_frames-1]=0


    def Dissonance(self):
        self.dissonance = np.zeros(self.n_frames)
        for b1 in range(self.n_bins):
            for b2 in range(self.n_bins):
                # Modèle de Sethares
                f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                freq = [f1, f2]
                freq.sort()
                s = S0 + (S1*freq[0] + 19)
                diss = np.exp(-B1*s*(freq[1]-freq[0]))-np.exp(-B2*s*(freq[1]-freq[0]))
                for p1 in range(self.n_pistes-1):
                    for p2 in range(p1+1, self.n_pistes-1):
                        self.dissonance = self.dissonance + (self.chromPistesSync[p1][b1,:] * self.chromPistesSync[p2][b2,:]) * diss
        self.dissonance[0]=0
        self.dissonance[self.n_frames-1]=0


    def DissonanceSignal(self):
        self.dissonanceSignal = np.zeros(self.n_frames)
        for b1 in range(self.n_bins-1):
            for b2 in range(b1+1,self.n_bins):
                # Modèle de Sethares
                f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                freq = [f1, f2]
                freq.sort()
                s = S0 + (S1*freq[0] + 19)
                diss = np.exp(-B1*s*(freq[1]-freq[0]))-np.exp(-B2*s*(freq[1]-freq[0]))

                self.dissonanceSignal = self.dissonanceSignal + (self.chromSync[b1,:] * self.chromSync[b2,:]) * diss
        self.dissonanceSignal[0]=0
        self.dissonanceSignal[self.n_frames-1]=0

    def Tension(self):
        self.tension = np.zeros(self.n_frames)
        for b1 in range(self.n_bins):
            for b2 in range(self.n_bins):
                for b3 in range(self.n_bins):

                    int = [abs(b3-b1), abs(b2-b1), abs(b3-b2)]
                    int.sort()
                    monInt = int[1]-int[0]
                    tens = np.exp(-((int[1]-int[0])* BINS_PER_OCTAVE/(12*δ))**2)
                    for p1 in range(self.n_pistes-2):
                        for p2 in range(p1+1, self.n_pistes-1):
                            for p3 in range(p2+1, self.n_pistes):
                                self.tension = self.tension + (self.chromPistesSync[p1][b1,:] * self.chromPistesSync[p2][b2,:] * self.chromPistesSync[p3][b3,:]) * tens
        self.tension[0]=0
        self.tension[self.n_frames-1]=0

        # Normalisation...


    def TensionSignal(self):
        self.tensionSignal = np.zeros(self.n_frames)
        for b1 in range(self.n_bins-2):
            for b2 in range(b1+1, self.n_bins-1):
                for b3 in range(b2+2, self.n_bins):
                    int = [abs(b3-b1), abs(b2-b1), abs(b3-b2)]
                    int.sort()
                    monInt = int[1]-int[0]
                    tens = np.exp(-((int[1]-int[0])* BINS_PER_OCTAVE/(12*δ))**2)

                    self.tensionSignal = self.tensionSignal + (self.chromSync[b1,:] * self.chromSync[b2,:] * self.chromSync[b3,:]) * tens
        self.tensionSignal[0]=0
        self.tensionSignal[self.n_frames-1]=0
        # Normalisation...


    def ComputeDescripteurs(self, space = ['concordance','concordanceTot']):
        """Calcule les descripteurs indiqués dans 'space', puis les affiche"""

        dim = len(space)
        if 'concordance' in space: self.Concordance()
        if 'concordanceTot' in space: self.ConcordanceTot()
        if 'tension' in space: self.Tension()
        if 'dissonance' in space: self.Dissonance()
        if 'tensionSignal' in space: self.TensionSignal()
        if 'dissonanceSignal' in space: self.DissonanceSignal()
        #print(np.shape(np.asarray(self.chromConc)))

        #Plot les spectrogrammes
        if plot_chromDescr:
            plt.figure(figsize=(13, 7))
            ax1 = plt.subplot(3,1,1)
            librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times)
            plt.title('CQT spectrogram')

            plt.subplot(3, 1, 2, sharex=ax1)
            librosa.display.specshow(librosa.amplitude_to_db(self.chromConc, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times)
            plt.title('Spectre de concordance')

            plt.subplot(3, 1, 3, sharex=ax1)
            librosa.display.specshow(librosa.amplitude_to_db(self.chromConcTot, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times)
            plt.title('Spectre de concordance Totale')
            plt.show()

        #Plot les descripteurs harmoniques
        if plot_descr:
            plt.figure(figsize=(13, 7))
            ax1 = plt.subplot(dim+1,1,1)
            librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times)
            plt.title('CQT spectrogram')

            for k, descr in enumerate(space):
                plt.subplot(dim+1, 1, k+2, sharex=ax1)
                plt.hlines(getattr(self, descr)[1:(self.n_frames-1)], self.onset_times[1:(self.n_frames-1)], self.onset_times[2:self.n_frames], color=['b','r','g','c','m','y'][k], label=descr)
                plt.vlines(self.onset_times[1:self.n_frames], 0, max(getattr(self, descr)), color='k', alpha=0.9, linestyle='--')
                plt.legend(frameon=True, framealpha=0.75)
            plt.show()




title = 'PalestrinaM'
#Palestrina, PalestrinaM
y, sr = librosa.load('/Users/manuel/Github/DescripteursHarmoniquesAudio/Exemples/'+title+'.wav')
y1, sr = librosa.load('/Users/manuel/Github/DescripteursHarmoniquesAudio/Exemples/'+title+'-Basse.wav')
y2, sr = librosa.load('/Users/manuel/Github/DescripteursHarmoniquesAudio/Exemples/'+title+'-Alto.wav')
y3, sr = librosa.load('/Users/manuel/Github/DescripteursHarmoniquesAudio/Exemples/'+title+'-Soprano.wav')
if params.SemiManual:
    delOnsets = getattr(params, 'delOnsets'+'_'+title)
    addOnsets = getattr(params, 'addOnsets'+'_'+title)
    α = getattr(params, 'paramsDetOnsets'+'_'+title)[0]
    β = getattr(params, 'paramsDetOnsets'+'_'+title)[1]
    H = getattr(params, 'paramsDetOnsets'+'_'+title)[2]
    T = getattr(params, 'paramsDetOnsets'+'_'+title)[3]
    T_att = getattr(params, 'paramsDetOnsets'+'_'+title)[4]


Notemin = 'D3'
Notemax = 'D9'

S = SignalSepare(y, sr, [y1,y2,y3], Notemin, Notemax, delOnsets, addOnsets)
S.DetectionOnsets()
S.Clustering()
#S.Concordance()
S.ComputeDescripteurs(space = [])
