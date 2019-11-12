from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter, attrgetter, truediv
import statistics as stat
from scipy import signal
import math
import matplotlib.image as mpimg

#from scipy import signal

import librosa
import librosa.display
import wave
import sys
import soundfile as sf
import os
import pyaudio
import threading
import pickle


import params

WINDOW = params.WINDOW
BINS_PER_OCTAVE = params.BINS_PER_OCTAVE
BINS_PER_OCTAVE_ONSETS = params.BINS_PER_OCTAVE_ONSETS
FILTER_SCALE = params.FILTER_SCALE
STEP = 512
α = params.α
ω = params.ω
H = params.H
T = params.T
T_att = params.T_att
cmap = params.cmap




class AudioFile:
    chunk = 1024

    def __init__(self, file):
        """ Init audio stream """
        self.file = file
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def __truePlay(self):
        data = self.wf.readframes(self.chunk)
        while data != '':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def play(self):
        """ Play entire file """
        x = threading.Thread(target=self.__truePlay, args=())
        x.start()


    def close(self):
        """ Graceful shutdown """
        self.stream.close()
        self.p.terminate()

class SignalSepare:
    """ Prend en entrée en signal et le signal des pistes audio séparées. """

    def __init__(self, signal, sr, pistes, Notemin  = 'D3', Notemax = 'D9',  onset_frames = [], delOnsets = [], addOnsets = [], score = [], instrument = ''):
        self.y = signal
        self.pistes = pistes
        self.sr = sr
        self.n_pistes = len(pistes)
        self.Notemin = Notemin
        self.Notemax = Notemax
        self.delOnsets = delOnsets
        self.addOnsets = addOnsets
        self.score = score
        self.instrument = instrument
        self.n_bins_ONSETS = 0
        self.n_bins = 0
        self.N = 0
        self.fmin = 0
        self.fmax = 0
        self.n_bins = 0
        self.n_att = 0
        self.n_frames = 0
        self.Dev = []
        self.Seuil = []
        self.times = []
        self.Onset_given = True
        self.onset_times = []
        self.onset_frames = onset_frames
        self.Percu = []
        self.Chrom_ONSETS = []
        self.ChromDB_ONSETS = []
        self.ChromSync_ONSETS = []
        self.ChromPistesSync_ONSETS = []
        self.chromSync = []
        self.chromSyncDB = []
        self.chromPistesSync = []
        self.chromSyncSimpl = []
        self.chromPistesSyncSimpl = []
        self.energy = []
        self.energyPistes = []
        self.activation =  []
        self.n_notes = []
        self.chrom_concordance = []
        self.concordance = []
        self.chrom_concordanceTot = []
        self.concordanceTot = []
        self.chrom_concordance3 = []
        self.concordance3 = []
        self.tension = []
        self.dissonance = []
        self.tensionSignal = []
        self.chrom_dissonance = []
        self.dissonanceSignal = []
        self.chrom_harmonicChange = []
        self.harmonicChange = []
        self.chrom_crossConcordance = []
        self.crossConcordance = []
        self.chrom_crossConcordanceTot = []
        self.crossConcordanceTot = []
        self.chrom_diffConcordance = []
        self.diffConcordance = []
        self.harmonicity = []
        self.virtualPitch = []



    def DetectionOnsets(self):
        self.fmin = librosa.note_to_hz(self.Notemin)
        self.fmax = librosa.note_to_hz(self.Notemax)
        #Nmin = int((sr/(fmax*(2**(1/BINS_PER_OCTAVE)-1))))
        #Nmax = int((sr/(fmin*(2**(1/BINS_PER_OCTAVE)-1))))
        self.n_bins_ONSETS = int((librosa.note_to_midi(self.Notemax) - librosa.note_to_midi(self.Notemin))*BINS_PER_OCTAVE_ONSETS/12)
        self.Chrom_ONSETS = np.abs(librosa.cqt(y=self.y, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE_ONSETS, n_bins=self.n_bins_ONSETS, window=WINDOW))
        self.ChromDB_ONSETS = librosa.amplitude_to_db(self.Chrom_ONSETS, ref=np.max)
        self.N = len(self.ChromDB_ONSETS[0])
        self.times = librosa.frames_to_time(np.arange(self.N), sr=sr, hop_length=STEP)

        # CALCUL DES ONSETS (pour onset précalculé, le rentrer dans self.onset_frames à l'initialisation)
        if len(self.onset_frames) == 0:
            self.Onset_given =  False
            Diff = np.zeros((self.n_bins_ONSETS,self.N))
            self.Dev = np.zeros(self.N)
            for j in range(1,self.N):
                for i in range(self.n_bins_ONSETS):
                    Diff[i,j] = np.abs(self.ChromDB_ONSETS[i,j]-self.ChromDB_ONSETS[i,j-1])
                    self.Dev[j] = sum(Diff[:,j])


            # FONCTION DE SEUIL
            # Ajout de zéros en queue et en tête
            l = []
            Onsets = []
            for k  in range(int(H/2)):
                l.append(0)
            for val in self.Dev:
                l.append(val)
            for k  in range(int(H/2)):
                l.append(0)
            #Calcul de la médiane
            for i in range(self.N):
                self.Seuil.append(α + ω*stat.median(l[i:i+H]))
                if self.Dev[i] > self.Seuil[i]:
                    Onsets.append(i)

            # FONCTION DE TRI SUR LES  ONSETS
            # Onsets espacés d'au moins T
            i=0
            while i<(len(Onsets)-1):
                while (i<(len(Onsets)-1)) and (self.times[Onsets[i+1]]< self.times[Onsets[i]]+T):
                    if (self.Dev[Onsets[i+1]]-self.Seuil[Onsets[i+1]]) < (self.Dev[Onsets[i]]-self.Seuil[Onsets[i]]): del Onsets[i+1]
                    #if (Dev[Onsets[i+1]]) < (Dev[Onsets[i]]): del Onsets[i+1]
                    else: del Onsets[i]
                i=i+1

            # Suppression manuelle des onsets en trop (cela nécessite d'avoir affiché les onsets jusqu'ici détectés)
            self.delOnsets.sort(reverse = True)
            for o in self.delOnsets:
                Onsets.pop(o-1)

             #Ajout manuel des onsets
            for t in self.addOnsets:
                Onsets.append(librosa.time_to_frames(t, sr=sr, hop_length=STEP))
                Onsets.sort()
            self.onset_frames = librosa.util.fix_frames(Onsets, x_min=0, x_max=self.ChromDB_ONSETS.shape[1]-1)


        self.onset_times = librosa.frames_to_time(self.onset_frames, sr=sr, hop_length = STEP)
        self.n_frames = len(self.onset_frames)-1



        # TRANSFORMÉE avec la précision due pour l'analyse
        self.n_bins = int((librosa.note_to_midi(self.Notemax) - librosa.note_to_midi(self.Notemin))*BINS_PER_OCTAVE/12)
        self.Chrom = np.abs(librosa.cqt(y=self.y, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins, window=WINDOW, filter_scale = FILTER_SCALE))
        # Décomposition partie harmonique / partie percussive
        if params.decompo_hpss: self.Chrom = librosa.decompose.hpss(self.Chrom, margin=params.margin)[0]
        self.ChromDB = librosa.amplitude_to_db(self.Chrom, ref=np.max)


        #Synchronisation sur les onsets, en enlevant le début et la fin des longues frames
        self.chromSync = np.zeros((self.n_bins,self.n_frames))
        self.n_att = int(librosa.time_to_frames(T_att, sr=self.sr, hop_length = STEP))
        for j in range(self.n_frames):
            for i in range(self.n_bins):
                self.chromSync[i,j] = np.median(self.Chrom[i][(self.onset_frames[j]+self.n_att):(self.onset_frames[j+1]-self.n_att)])
        self.chromSyncDB = librosa.amplitude_to_db(self.chromSync, ref=np.max)

        #Calcul de l'énergie
        for j in range(self.n_frames):
            self.energy.append(np.sum(np.multiply(self.chromSync[:,j], self.chromSync[:,j])))




    def Clustering(self):
        """ Découpe et synchronise les pistes séparées sur les ONSETS, stoque le spectrogramme
        synchronisé en construisant self.chromPistesSync"""

        #  Construction de chromPistesSync
        ChromPistes = []
        for k, voice in enumerate(self.pistes):
            if params.decompo_hpss:
                ChromPistes.append(librosa.decompose.hpss(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)), margin=params.margin)[0])
            else: ChromPistes.append(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)))

            self.chromPistesSync.append(np.zeros((self.n_bins,self.n_frames)))
            for j in range(self.n_frames):
                for i in range(self.n_bins):
                    self.chromPistesSync[k][i,j] = np.median(ChromPistes[k][i][(self.onset_frames[j]+self.n_att):(self.onset_frames[j+1]-self.n_att)])

        # Calcul de l'énergie des pistes
        self.energyPistes = np.zeros((self.n_pistes, self.n_frames))
        for t in range(self.n_frames):
            for k in range(self.n_pistes):
                self.energyPistes[k,t] = np.sum(np.multiply(self.chromPistesSync[k][:,t], self.chromPistesSync[k][:,t]))

        # Calcul de la matrice d'activation (pour savoir quelles voix contiennent des notes à quel moment) + mise à zéro des pistes sans note
        max_energy = np.amax(self.energyPistes)
        self.activation = np.ones((self.n_pistes, self.n_frames))
        for k in range(self.n_pistes):
            for t in range(self.n_frames):
                if (self.energyPistes[k,t] < params.seuil_activation * max_energy):
                    self.activation[k,t] = 0
                    self.chromPistesSync[k][:,t] = 0

        # Calcul du nombre de notes
        self.n_notes = np.sum(self.activation, axis=0)



    def SimplifySpectrum(self):
        self.chromSyncSimpl = np.copy(self.chromSync)
        self.chromPistesSyncSimpl= np.copy(self.chromPistesSync)
        for t in range(self.n_frames):
            for i in range(1, self.n_bins - 1):
                # Simplification de self.chromSync
                if not self.chromSync[i-1,t]<self.chromSync[i,t]>self.chromSync[i+1,t]:
                    self.chromSyncSimpl[i-1,t] = 0
                # Simplification de self.chromPistesSync
                for k in range(len(self.pistes)):
                    if not self.chromPistesSync[k][i-1,t]<self.chromPistesSync[k][i,t]>self.chromPistesSync[k][i+1,t]:
                        self.chromPistesSyncSimpl[k][i-1,t] = 0


    def Concordance(self):
        """Multiplie les spectres (cqt) des différentes pistes pour créer le spectre de concordance,
        et calcule la concordance en sommant sur les fréquences"""

        self.chrom_concordance = np.zeros((self.n_bins,self.n_frames))
        for k in range(self.n_pistes-1):
            for l in range(k+1, self.n_pistes):
                if params.norm_conc == 'None':
                    self.chrom_concordance = self.chrom_concordance + np.multiply(self.chromPistesSync[k], self.chromPistesSync[l])
                if params.norm_conc == 'note_by_note':
                    self.chrom_concordance = self.chrom_concordance + np.divide(np.multiply(self.chromPistesSync[k], self.chromPistesSync[l]), np.sqrt(np.multiply(self.energyPistes[k], self.energyPistes[l])))
                elif params.norm_conc == 'chord_by_chord':
                    self.chrom_concordance = self.chrom_concordance + np.divide(np.multiply(self.chromPistesSync[k], self.chromPistesSync[l]), self.energy)

        # Normalisation en fonction du nombre de notes
        for t in range(self.n_frames):
            if self.n_notes[t]>=2:
                self.chrom_concordance[:,t] = self.chrom_concordance[:,t]/(self.n_notes[t]*(self.n_notes[t]-1)/2)

        self.concordance = self.chrom_concordance.sum(axis=0)
        self.concordance[0]=0
        self.concordance[self.n_frames-1]=0


    def ConcordanceTot(self):
        """Multiplie les spectres (cqt) des différentes pistes pour créer le spectre de concordance,
        et calcule la concordance en sommant sur les fréquences"""

        self.chrom_concordanceTot = np.ones((self.n_bins,self.n_frames))
        for t in range(self.n_frames):
            for k in range(self.n_pistes):
                if self.activation[k,t]:
                    self.chrom_concordanceTot[:,t] = np.multiply(self.chrom_concordanceTot[:,t], self.chromPistesSync[k][:,t])
            if self.n_notes[t]>=1:
                self.chrom_concordanceTot[:,t] = np.divide(np.power(self.chrom_concordanceTot[:,t], 2/self.n_notes[t]), LA.norm(self.chromSync[:,t], self.n_notes[t])**2)
            self.concordanceTot.append(self.chrom_concordanceTot[:,t].sum(axis=0))

        # Normalisation...
        self.concordanceTot[0]=0
        self.concordanceTot[self.n_frames-1]=0


    def Concordance3(self):
        self.chrom_concordance3 = np.zeros((self.n_bins,self.n_frames))
        if self.n_pistes >= 3:
            for k in range(self.n_pistes-2):
                for l in range(k+1, self.n_pistes-1):
                    for m in range(l+1, self.n_pistes):
                        if params.norm_conc3 == 'None':
                            self.chrom_concordance3 = self.chrom_concordance3 + np.power(np.multiply(np.multiply(self.chromPistesSync[k], self.chromPistesSync[l]), self.chromPistesSync[m]),2/3)
                        if params.norm_conc3 == 'energy':
                            self.chrom_concordance3 = self.chrom_concordance3 + np.divide(np.power(np.multiply(np.multiply(self.chromPistesSync[k], self.chromPistesSync[l]), self.chromPistesSync[m]),2/3), self.energy)
                        elif params.norm_conc3 == 'norme3':
                            self.chrom_concordance3 = self.chrom_concordance3 + np.divide(np.power(np.multiply(np.multiply(self.chromPistesSync[k], self.chromPistesSync[l]), self.chromPistesSync[m]),2/3), np.power(LA.norm(self.chromSync,ord=3,axis=0),2))

            # Normalisation en fonction du nombre de notes
            for t in range(self.n_frames):
                if self.n_notes[t]>=3:
                    self.chrom_concordance3[:,t] = self.chrom_concordance3[:,t]/(self.n_notes[t]*(self.n_notes[t]-1)*(self.n_notes[t]-2)/6)

        self.concordance3 = self.chrom_concordance3.sum(axis=0)
        self.concordance3[0]=0
        self.concordance3[self.n_frames-1]=0








    def Dissonance(self):
        #fonction qui convertit les amplitudes en sonies
        # def Sones(Ampl):
        #     P = Ampl/np.sqrt(2)
        #     SPL = 20*np.log10(P/params.P_ref)
        #     return ((1/16)*np.power(2,SPL/10))
        self.chrom_dissonance = np.zeros((self.n_bins,self.n_frames))
        self.dissonance = np.zeros(self.n_frames)
        for b1 in range(self.n_bins):
            for b2 in range(self.n_bins):
                # Modèle de Sethares
                f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                freq = [f1, f2]
                freq.sort()
                s = 0.44*(np.log(params.β2/params.β1)/(params.β2-params.β1))*(freq[1]-freq[0])/(freq[0]**(0.477))
                diss = np.exp(-params.β1*s)-np.exp(-params.β2*s)
                for p1 in range(self.n_pistes-1):
                    for p2 in range(p1+1, self.n_pistes):
                        if params.spectrDiss_Simpl:
                            if params.type_diss == 'produit':
                                self.chrom_dissonance[b1,:] = self.chrom_dissonance[b1,:] + (self.chromPistesSyncSimpl[p1][b1,:] * self.chromPistesSyncSimpl[p2][b2,:]) * diss/2
                                self.chrom_dissonance[b2,:] = self.chrom_dissonance[b2,:] + (self.chromPistesSyncSimpl[p1][b1,:] * self.chromPistesSyncSimpl[p2][b2,:]) * diss/2
                            elif params.type_diss == 'minimum':
                                self.chrom_dissonance[b1,:] = self.chrom_dissonance[b1,:] + np.fmin(self.chromPistesSyncSimpl[p1][b1,:], self.chromPistesSyncSimpl[p2][b2,:]) * diss/2
                                self.chrom_dissonance[b2,:] = self.chrom_dissonance[b2,:] + np.fmin(self.chromPistesSyncSimpl[p1][b1,:], self.chromPistesSyncSimpl[p2][b2,:]) * diss/2
                        else:
                            if params.type_diss == 'produit':
                                self.chrom_dissonance[b1,:] = self.chrom_dissonance[b1,:] + (self.chromPistesSync[p1][b1,:] * self.chromPistesSync[p2][b2,:]) * diss/2
                                self.chrom_dissonance[b2,:] = self.chrom_dissonance[b2,:] + (self.chromPistesSync[p1][b1,:] * self.chromPistesSync[p2][b2,:]) * diss/2
                            elif params.type_diss == 'minimum':
                                self.chrom_dissonance[b1,:] = self.chrom_dissonance[b1,:] + np.fmin(self.chromPistesSync[p1][b1,:], self.chromPistesSync[p2][b2,:]) * diss/2
                                self.chrom_dissonance[b2,:] = self.chrom_dissonance[b2,:] + np.fmin(self.chromPistesSync[p1][b1,:], self.chromPistesSync[p2][b2,:]) * diss/2

        # Normalisation en fonction du nombre de notes
        for t in range(self.n_frames):
            if self.n_notes[t]>=2:
                self.chrom_dissonance[:,t] = self.chrom_dissonance[:,t]/(self.n_notes[t]*(self.n_notes[t]-1)/2)

        if params.norm_diss:
            if params.type_diss == 'produit': norm = self.energy
            else: norm = np.sqrt(self.energy)
            self.dissonance = np.divide(self.chrom_dissonance.sum(axis=0),norm)
        else : self.dissonance = self.chrom_dissonance.sum(axis=0)

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
                s = 0.44*(np.log(params.β2/params.β1)/(params.β2-params.β1))*(freq[1]-freq[0])/(freq[0]**(0.477))
                diss = np.exp(-params.β1*s)-np.exp(-params.β2*s)
                if params.type_diss == 'produit':
                    self.dissonanceSignal = self.dissonanceSignal + (self.chromSync[b1,:] * self.chromSync[b2,:]) * diss
                elif params.type_diss == 'minimum':
                    self.dissonanceSignal = self.dissonanceSignal + np.fmin(self.chromSync[b1,:], self.chromSync[b2,:]) * diss
                # self.dissonanceSignal = self.dissonanceSignal + (self.chromSync[b1,:] * self.chromSync[b2,:]) * diss
        if params.norm_diss: self.dissonanceSignal = np.divide(self.dissonanceSignal, self.energy)
        self.dissonanceSignal[0]=0
        self.dissonanceSignal[self.n_frames-1]=0


    def Tension(self):
        #Calcul du spectre du signal en 1/4 de tons pour avoir un calcul de la tension en temps raisonnable
        ChromPistes_ONSETS = []
        for k, voice in enumerate(self.pistes):
            ChromPistes_ONSETS.append(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE_ONSETS, n_bins=self.n_bins_ONSETS)))
            self.ChromPistesSync_ONSETS.append(np.zeros((self.n_bins_ONSETS,self.n_frames)))
            for j in range(self.n_frames):
                for i in range(self.n_bins_ONSETS):
                    self.ChromPistesSync_ONSETS[k][i,j] = np.median(ChromPistes_ONSETS[k][i][(self.onset_frames[j]+self.n_att):(self.onset_frames[j+1]-self.n_att)])

        #Clacul tension
        self.tension = np.zeros(self.n_frames)
        for b1 in range(self.n_bins_ONSETS):
            for b2 in range(self.n_bins_ONSETS):
                for b3 in range(self.n_bins_ONSETS):

                    int = [abs(b3-b1), abs(b2-b1), abs(b3-b2)]
                    int.sort()
                    monInt = int[1]-int[0]
                    if monInt == 0: tens = 1
                    # elif monInt == 1 : tens = 0.5
                    else: tens = 0
                    # tens = np.exp(-((int[1]-int[0])* BINS_PER_OCTAVE/(12*params.δ))**2)
                    for p1 in range(self.n_pistes-2):
                        for p2 in range(p1+1, self.n_pistes-1):
                            for p3 in range(p2+1, self.n_pistes):
                                self.tension = self.tension + (self.ChromPistesSync_ONSETS[p1][b1,:] * self.ChromPistesSync_ONSETS[p2][b2,:] * self.ChromPistesSync_ONSETS[p3][b3,:]) * tens
        self.tension = np.divide(self.tension, np.power(self.energy, 3/2))
        self.tension[0]=0
        self.tension[self.n_frames-1]=0


    def TensionSignal(self):
        # Calcul des spectres des pistes en 1/4 de tons pour avoir un calcul de la tensionSignal en temps raisonnable
        self.ChromSync_ONSETS = np.zeros((self.n_bins_ONSETS,self.n_frames))
        for j in range(self.n_frames):
            for i in range(self.n_bins_ONSETS):
                self.ChromSync_ONSETS[i,j] = np.median(self.Chrom_ONSETS[i][(self.onset_frames[j]+self.n_att):(self.onset_frames[j+1]-self.n_att)])

        #Calcul tensionSignal
        self.tensionSignal = np.zeros(self.n_frames)
        for b1 in range(self.n_bins_ONSETS-2):
            for b2 in range(b1+1, self.n_bins_ONSETS-1):
                for b3 in range(b2+2, self.n_bins_ONSETS):
                    int = [abs(b3-b1), abs(b2-b1), abs(b3-b2)]
                    int.sort()
                    monInt = int[1]-int[0]
                    tens = np.exp(-((int[1]-int[0])* BINS_PER_OCTAVE/(12*params.δ))**2)
                    self.tensionSignal = self.tensionSignal + (self.ChromSync_ONSETS[b1,:] * self.ChromSync_ONSETS[b2,:] * self.ChromSync_ONSETS[b3,:]) * tens
        self.tensionSignal = np.divide(self.tensionSignal, np.power(self.energy, 3/2))
        self.tensionSignal[0]=0
        self.tensionSignal[self.n_frames-1]=0


    def CrossConcordance(self):
        if len(self.concordance) == 0: self.Concordance()

        self.chrom_crossConcordance = np.zeros((self.n_bins,self.n_frames-1))
        for t in range(self.n_frames-1):
            self.chrom_crossConcordance[:,t] = np.multiply(self.chrom_concordance[:,t],self.chrom_concordance[:,t+1])
            if params.norm_crossConc == 'energy + conc':
                if self.concordance[t]*self.concordance[t+1]!=0:
                    self.chrom_crossConcordance[:,t] = np.divide(self.chrom_crossConcordance[:,t], self.concordance[t]*self.concordance[t+1])
        self.crossConcordance = self.chrom_crossConcordance.sum(axis=0)
        self.crossConcordance[0]=0
        self.crossConcordance[self.n_frames-2]=0


    def CrossConcordanceTot(self):
        if len(self.concordanceTot) == 0: self.ConcordanceTot()

        self.chrom_crossConcordanceTot = np.zeros((self.n_bins,self.n_frames-1))
        for t in range(self.n_frames-1):
            self.chrom_crossConcordanceTot[:,t] = np.multiply(self.chrom_concordanceTot[:,t],self.chrom_concordanceTot[:,t+1])
            if params.norm_crossConcTot == 'energy + concTot':
                if self.concordanceTot[t]*self.concordanceTot[t+1]!=0:
                    self.chrom_crossConcordanceTot[:,t] = np.divide(self.chrom_crossConcordanceTot[:,t], self.concordanceTot[t]*self.concordanceTot[t+1])
        self.crossConcordanceTot = self.chrom_crossConcordanceTot.sum(axis=0)
        self.crossConcordanceTot[0]=0
        self.crossConcordanceTot[self.n_frames-2]=0


    def HarmonicChange(self):
        self.chrom_harmonicChange = np.zeros((self.n_bins,self.n_frames-1))
        for t in range(self.n_frames-1):
            # Par défaut, params.type_harmChange == 'relative'


            if params.norm_harmChange == 'None':
                self.chrom_harmonicChange[:,t] = self.chromSync[:,t+1] - self.chromSync[:,t]

            elif params.norm_harmChange == 'frame_by_frame':
                self.chrom_harmonicChange[:,t] = np.subtract([x/np.sqrt(self.energy[t+1]) for x in self.chromSync[:,t+1]], [y/np.sqrt(self.energy[t]) for y in self.chromSync[:,t]])

            elif params.norm_harmChange == 'general':
                self.chrom_harmonicChange[:,t] = np.divide(self.chromSync[:,t+1] - self.chromSync[:,t], (self.energy[t+1]*self.energy[t])**(1/4))

            if params.type_harmChange == 'absolute': self.chrom_harmonicChange[:,t] = np.abs(self.chrom_harmonicChange[:,t])

        self.harmonicChange = self.chrom_harmonicChange.sum(axis=0)
        self.harmonicChange[0]=0
        self.harmonicChange[self.n_frames-2]=0


    def DiffConcordance(self):
        self.chrom_diffConcordance = np.zeros((self.n_bins,self.n_frames-1))
        if params.norm_diffConc == 'chord_by_chord':
            for t in range(self.n_frames-1):
                self.chrom_diffConcordance[:,t] = np.multiply(self.chromSync[:,t], self.chromSync[:,t+1])
                self.chrom_diffConcordance[:,t] = np.divide(self.chrom_diffConcordance[:,t], np.sqrt(self.energy[t] * self.energy[t+1]))
                if self.n_notes[t]*self.n_notes[t+1] >= 1:
                    self.chrom_diffConcordance[:,t] = self.chrom_diffConcordance[:,t]/(self.n_notes[t]*self.n_notes[t+1])
        elif params.norm_diffConc == 'note_by_note':
            for k in range(self.n_pistes):
                for l in range(self.n_pistes):
                    for t in range(self.n_frames-1):
                        self.chrom_diffConcordance[:,t] = self.chrom_diffConcordance[:,t] + np.divide(np.multiply(self.chromPistesSync[k][:,t], self.chromPistesSync[l][:,t+1]), np.sqrt(self.energyPistes[k][t] * self.energyPistes[l][t+1]))
                        if self.n_notes[t]*self.n_notes[t+1] >= 1:
                            self.chrom_diffConcordance[:,t] = self.chrom_diffConcordance[:,t]/(self.n_notes[t]*self.n_notes[t+1])

        self.diffConcordance = self.chrom_diffConcordance.sum(axis=0)
        self.diffConcordance[0]=0
        self.diffConcordance[self.n_frames-2]=0


    def Harmonicity(self):
        # Construction du spectre harmonique
        dec = BINS_PER_OCTAVE/6 # décalage d'un ton pour tenir compte de l'épaisseur des gaussiennes
        epaiss = int(np.rint(BINS_PER_OCTAVE/(2*params.σ)))
        SpecHarm = np.zeros(2*int(dec) + int(np.rint(BINS_PER_OCTAVE * np.log2(params.κ))))
        for k in range(params.κ):
            pic =  int(dec + np.rint(BINS_PER_OCTAVE * np.log2(k+1)))
            for i in range(-epaiss, epaiss+1):
                SpecHarm[pic + i] = 1/(k+1)**params.decr


        # Correlation avec le spectre réel
        for t in range(self.n_frames):
            self.harmonicity.append(max(np.correlate(np.power(self.chromSync[:,t],params.norm_harm), SpecHarm,"full")) / self.energy[t]**(params.norm_harm/2))

            # Virtual Pitch
            self.virtualPitch.append(np.argmax(np.correlate(np.power(self.chromSync[:,t],2), SpecHarm,"full")))
        virtualNotes = librosa.hz_to_note([self.fmin * (2**((i-len(SpecHarm)+dec+1)/BINS_PER_OCTAVE)) for i in self.virtualPitch] , cents = False)
        print(virtualNotes[1:self.n_frames-1])

        print(epaiss)






    def ComputeDescripteurs(self, space = ['concordance','concordanceTot']):
        """Calcule les descripteurs indiqués dans 'space', puis les affiche"""

        dim = len(space)
        if 'concordance' in space: self.Concordance()
        if 'concordanceTot' in space: self.ConcordanceTot()
        if 'concordance3' in space: self.Concordance3()
        if 'tension' in space: self.Tension()
        if 'dissonance' in space: self.Dissonance()
        if 'tensionSignal' in space: self.TensionSignal()
        if 'dissonanceSignal' in space: self.DissonanceSignal()
        if 'harmonicity' in space: self.Harmonicity()
        if 'crossConcordance' in space: self.CrossConcordance()
        if 'crossConcordanceTot' in space: self.CrossConcordanceTot()
        if 'harmonicChange' in space: self.HarmonicChange()
        if 'diffConcordance' in space: self.DiffConcordance()


        #Plot des représentations symboliques
        if params.plot_symb:
            if params.play:
                # Supprimer le  fichier stereo_file.wav s'il existe
                if os.path.exists("stereo_file.wav"):
                  os.remove("stereo_file.wav")
                # Écrire un fichier wav à partir de y et sr
                sf.write('stereo_file.wav', self.y, self.sr)
                a = AudioFile("stereo_file.wav")


            if (dim==2):
                fig, ax = plt.subplots()
                xdata, ydata = [], []
                xdescr, ydescr = getattr(self, space[0]), getattr(self, space[1])
                ln, = plt.plot([], [], 'r'+'--'+'o')

                def init():
                    ax.set_xlim(min(xdescr[1:self.n_frames-1]), max(xdescr[1:self.n_frames-1])*6/5)
                    ax.set_ylim(min(ydescr[1:self.n_frames-1]), max(ydescr[1:self.n_frames-1])*6/5)
                    if params.play : a.play()
                    return ln,

                def update(time):
                    if (self.onset_times[0] <= time < self.onset_times[1]): pass
                    elif (self.onset_times[self.n_frames-1] <= time <= self.onset_times[self.n_frames]): pass
                    else:
                        i = 1
                        found = False
                        while (not found):
                            if (self.onset_times[i] <= time <= self.onset_times[i+1]):
                                xdata.append(xdescr[i])
                                ydata.append(ydescr[i])
                                ln.set_data(xdata, ydata)
                                found = True
                            else: i=i+1


                    #ax.annotate(frame+1, (xdescr[frame], ydescr[frame]))
                    return ln,

                #a.play()
                #a.close()
                ani = FuncAnimation(fig, update, frames=self.times, init_func=init, blit=True, interval=1000*STEP/self.sr, repeat=False)
                # a.close()
                #ani = FuncAnimation(fig, update, frames=self.times, init_func=init, blit=True, interval=23 , repeat=False)
                plt.xlabel(space[0])
                plt.ylabel(space[1])
                plt.show()

    def Affichage(self, space = ['concordance', 'concordanceTot']):
        #Plot de la recherche d'ONSETS
        if params.plot_onsets:
            plt.figure(1,figsize=(13, 7))
            ax1 = plt.subplot(3, 1, 1)
            librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times,cmap=cmap)
            plt.title('CQT spectrogram')

            plt.subplot(3, 1, 2, sharex=ax1)
            if not self.Onset_given:
                plt.plot(self.times, self.Dev, label='Deviation')
                plt.plot(self.times, self.Seuil, color='g', label='Seuil')
                plt.vlines(self.times[self.onset_frames[1:len(self.onset_frames)-1]], 0, self.Dev.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
            else:
                plt.vlines(self.times[self.onset_frames], 0, 1, color='r', alpha=0.9, linestyle='--', label='Onsets')

            plt.axis('tight')
            plt.legend(frameon=True, framealpha=0.75)

            plt.subplot(3, 1, 3, sharex=ax1)
            librosa.display.specshow(self.chromSyncDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times, cmap=cmap)
            plt.tight_layout()

        #Plot de la décomposition en partie harmonique / partie percussive
        if params.plot_decompo_hpss & params.decompo_hpss:
            plt.figure(2,figsize=(13, 7))
            plt.subplot(3, 1, 1)
            Chrom = np.abs(librosa.cqt(y=self.y, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins, window=WINDOW, filter_scale = FILTER_SCALE))
            librosa.display.specshow(librosa.amplitude_to_db(Chrom, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
            plt.title('Full cqt transform')

            plt.subplot(3, 1, 2)
            librosa.display.specshow(librosa.amplitude_to_db(self.Chrom,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
            plt.title('Harmonic part')

            plt.subplot(3, 1, 3)
            librosa.display.specshow(librosa.amplitude_to_db(Chrom - self.Chrom,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
            plt.title('Percussive part')
            plt.tight_layout()

        #Plot des pistes
        if params.plot_pistes:
            plt.figure(3,figsize=(13, 7))
            ax1 = plt.subplot(self.n_pistes,1,1)
            librosa.display.specshow(librosa.amplitude_to_db(self.chromPistesSync[0], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times,cmap=cmap)
            for k in range(1, self.n_pistes):
                plt.subplot(self.n_pistes, 1, k+1, sharex=ax1)
                librosa.display.specshow(librosa.amplitude_to_db(self.chromPistesSync[k], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times,cmap=cmap)
            plt.tight_layout()

        #Plot des spectres simplifiés
        if params.plot_partiels:
            plt.figure(4,figsize=(13, 7))
            ax1 = plt.subplot(3,1,1)
            librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times, cmap=cmap)
            plt.title('CQT spectrogram')

            plt.subplot(3, 1, 2, sharex=ax1)
            librosa.display.specshow(librosa.amplitude_to_db(self.chromSync, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times, cmap=cmap)
            plt.title('Spectre syncronisé')

            plt.subplot(3, 1, 3, sharex=ax1)
            librosa.display.specshow(librosa.amplitude_to_db(self.chromSyncSimpl, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times, cmap=cmap)
            plt.title('Partiels')
            plt.tight_layout()

        #Plot les spectrogrammes
        if params.plot_chromDescr:

            #Construction de la liste des descripteurs avec Chrom
            spaceChrom = []
            for descr in space:
                if descr in ['concordance','concordance3','concordanceTot','dissonance','crossConcordance','crossConcordanceTot','harmonicChange','diffConcordance']: spaceChrom.append(descr)


            dimChrom = len(space)
            times_plotChromDyn = [self.onset_times[0]] + [t-0.25 for t in self.onset_times[2:self.n_frames-1]] + [t+0.25 for t in self.onset_times[2:self.n_frames-1]] + [self.onset_times[self.n_frames]]
            times_plotChromDyn.sort()

            plt.figure(5,figsize=(13, 7))
            # Partition
            if params.plot_score & (len(self.score)!=0):
                #plt.subplot(dim+1+s,1,s)
                img=mpimg.imread(self.score)
                score = plt.subplot(dimChrom+1,1,1)
                plt.axis('off')
                score.imshow(img)
                plt.title(title +' '+instrument)

            else:
                ax1 = plt.subplot(dimChrom+1,1,1)
                librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times, cmap=cmap)
                plt.title(title +' '+instrument)

            for k, descr in enumerate(spaceChrom):
                if (k==0) & params.plot_score & (len(self.score)!=0):
                    ax1 = plt.subplot(dimChrom+1,1,2)
                else: plt.subplot(dimChrom+1, 1, k+2, sharex=ax1)

                    # Descripteurs statiques
                if len(getattr(self, descr)) == self.n_frames:
                    if descr in ['dissonance']:
                        librosa.display.specshow(getattr(self, 'chrom_'+descr), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times, cmap=cmap)
                    else:
                        librosa.display.specshow(librosa.amplitude_to_db(getattr(self, 'chrom_'+descr), ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times, cmap=cmap)

                    # Descripteurs dynamiques
                else:
                    Max  = np.amax(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2])
                    librosa.display.specshow(librosa.amplitude_to_db(np.insert(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2]/Max, range(self.n_frames-2),1, axis=1), ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=np.asarray(times_plotChromDyn), cmap=cmap)
                plt.title('Spectre de ' + descr)

            # plt.tight_layout()


        #Plot les descripteurs harmoniques
        if params.plot_descr:
            dim = len(space)
            plt.figure(6,figsize=(13, 7))

            # Partition
            if params.plot_score & (len(self.score)!=0):
                #plt.subplot(dim+1+s,1,s)
                img=mpimg.imread(self.score)
                score = plt.subplot(dim+1,1,1)
                plt.axis('off')
                score.imshow(img)
                plt.title(title +' '+ instrument)

            else:
                ax1 = plt.subplot(dim+1,1,1)
                librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times,cmap=cmap)
                plt.title(title +' '+ instrument)

            for k, descr in enumerate(space):
                if (k==0) & params.plot_score & (len(self.score)!=0):
                    ax1 = plt.subplot(dim+1,1,2)
                else: plt.subplot(dim+1, 1, k+2, sharex=ax1)
                plt.vlines(self.onset_times[1:self.n_frames], min(getattr(self, descr)), max(getattr(self, descr)), color='k', alpha=0.9, linestyle='--')
                if not all(x>=0 for x in getattr(self, descr)):
                    plt.hlines(0,self.onset_times[0], self.onset_times[self.n_frames], alpha=0.5, linestyle = ':')

                # Descripteurs statiques
                if len(getattr(self, descr)) == self.n_frames:
                    plt.hlines(getattr(self, descr)[1:(self.n_frames-1)], self.onset_times[1:(self.n_frames-1)], self.onset_times[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][k], label=descr[0].upper() + descr[1:])
                # Descripteurs dynamiques
                elif len(getattr(self, descr)) == (self.n_frames-1):
                    plt.plot(self.onset_times[2:(self.n_frames-1)], getattr(self, descr)[1:(self.n_frames-2)],['b','r','g','c','m','y','b','r','g'][k]+'o', label=(descr[0].upper() + descr[1:]))
                    plt.hlines(getattr(self, descr)[1:(self.n_frames-2)], [t-0.25 for t in self.onset_times[2:(self.n_frames-1)]], [t+0.25 for t in self.onset_times[2:(self.n_frames-1)]], color=['b','r','g','c','m','y','b','r','g'][k], alpha=0.9, linestyle=':'  )
                plt.legend(frameon=True, framealpha=0.75)

            # plt.tight_layout()

        if params.plot_OneDescr:
            descr = space[0]
            plt.figure(7,figsize=(13, 7))

            # Partition
            if params.plot_score & (len(self.score)!=0):
                #plt.subplot(dim+1+s,1,s)
                img=mpimg.imread(self.score)
                score = plt.subplot(3,1,1)
                plt.axis('off')
                score.imshow(img)
                plt.title(title +' '+instrument)

            else:
                ax1 = plt.subplot(3,1,1)
                librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times,cmap=cmap)
                plt.title(title +' '+instrument)

            # Plot Spectre Descr
            if params.plot_score & (len(self.score)!=0):
                ax1 = plt.subplot(3,1,2)
            else: plt.subplot(3, 1, 2, sharex=ax1)

                # Descripteurs statiques
            if len(getattr(self, descr)) == self.n_frames:
                if descr in ['dissonance']:
                    librosa.display.specshow(getattr(self, 'chrom_'+descr), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times, cmap=cmap)
                else:
                    librosa.display.specshow(librosa.amplitude_to_db(getattr(self, 'chrom_'+descr), ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times, cmap=cmap)

                # Descripteurs dynamiques
            else:
                times_plotChromDyn = [self.onset_times[0]] + [t-0.25 for t in self.onset_times[2:self.n_frames-1]] + [t+0.25 for t in self.onset_times[2:self.n_frames-1]] + [self.onset_times[self.n_frames]]
                times_plotChromDyn.sort()
                Max  = np.amax(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2])
                librosa.display.specshow(librosa.amplitude_to_db(np.insert(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2]/Max, range(self.n_frames-2),1, axis=1), ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=np.asarray(times_plotChromDyn),cmap=cmap)
            plt.title('Spectre de ' + descr)

            # Plot Descr
            plt.subplot(3, 1, 3, sharex=ax1)
            plt.vlines(self.onset_times[1:self.n_frames], min(getattr(self, descr)), max(getattr(self, descr)), color='k', alpha=0.9, linestyle='--')
            if not all(x>=0 for x in getattr(self, descr)):
                plt.hlines(0,self.onset_times[0], self.onset_times[self.n_frames], alpha=0.5, linestyle = ':')

                # Descripteurs statiques
            if len(getattr(self, descr)) == self.n_frames:
                plt.hlines(getattr(self, descr)[1:(self.n_frames-1)], self.onset_times[1:(self.n_frames-1)], self.onset_times[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][1], label=descr)
                # Descripteurs dynamiques
            elif len(getattr(self, descr)) == (self.n_frames-1):
                plt.plot(self.onset_times[2:(self.n_frames-1)], getattr(self, descr)[1:(self.n_frames-2)],['b','r','g','c','m','y','b','r','g'][1]+'o', label=descr)
                plt.hlines(getattr(self, descr)[1:(self.n_frames-2)], [t-0.25 for t in self.onset_times[2:(self.n_frames-1)]], [t+0.25 for t in self.onset_times[2:(self.n_frames-1)]], color=['b','r','g','c','m','y','b','r','g'][0], alpha=0.9, linestyle=':'  )
            plt.legend(frameon=True, framealpha=0.75)
            plt.tight_layout()

        #Plot représentations abstraites
        if params.plot_abstr:
            if len(space)==2 :
                color = params.color_abstr
                l1 = getattr(self, space[0])[1:self.n_frames-1]
                l2 = getattr(self, space[1])[1:self.n_frames-1]
                plt.figure(8)
                ax = plt.subplot()
                if params.link_abstr: plt.plot(l1, l2, color+'--')
                plt.plot(l1, l2, color+'o')
                for i in range(len(l1)):
                    ax.annotate(i+1, (l1[i], l2[i]))
                plt.xlabel(space[0][0].upper() + space[0][1:])
                plt.ylabel(space[1][0].upper() + space[1][1:])
                # plt.title(title)

            else:
                color = params.color_abstr
                l1 = getattr(self, space[0])[1:self.n_frames-1]
                l2 = getattr(self, space[1])[1:self.n_frames-1]
                l3 = getattr(self, space[2])[1:self.n_frames-1]
                fig = plt.figure(9)
                ax = fig.add_subplot(111, projection='3d')
                if params.link_abstr: plt.plot(l1, l2, l3, color+'--')
                for i in range(len(l1)):
                    ax.scatter(l1[i], l2[i], l3[i], c=color, marker='o')
                    ax.text(l1[i], l2[i], l3[i], i+1, color='red')
                ax.set_xlabel(space[0][0].upper() + space[0][1:])
                ax.set_ylabel(space[1][0].upper() + space[1][1:])
                ax.set_zlabel(space[2][0].upper() + space[2][1:])
                # ax.set_title('title')


        plt.show()





title = 'SuiteAccordsPiano'
instrument = 'Piano'
#Palestrina, PalestrinaM, SuiteAccords, AccordsParalleles, 'SuiteAccordsOrgue', 'SuiteAccordsPiano', 'CadenceM','CadenceM2','AccordsM', 'SuiteAccordsViolin'
y, sr = librosa.load('Exemples/'+title+'.wav')
y1, sr = librosa.load('Exemples/'+title+'-Basse.wav')
y2, sr = librosa.load('Exemples/'+title+'-Alto.wav')
y3, sr = librosa.load('Exemples/'+title+'-Soprano.wav')
# y4, sr = librosa.load('Exemples/'+title+'-Tenor.wav')
delOnsets = []
addOnsets = []
if params.SemiManual:
    delOnsets = getattr(params, 'delOnsets'+'_'+title)
    addOnsets = getattr(params, 'addOnsets'+'_'+title)
    α = getattr(params, 'paramsDetOnsets'+'_'+title)[0]
    ω = getattr(params, 'paramsDetOnsets'+'_'+title)[1]
    H = getattr(params, 'paramsDetOnsets'+'_'+title)[2]
    T = getattr(params, 'paramsDetOnsets'+'_'+title)[3]
    T_att = getattr(params, 'paramsDetOnsets'+'_'+title)[4]



Notemin = 'D3' #'SuiteAccordsOrgue': A2
Notemax = 'D9'
score = 'Exemples/'+ title +'-score.png'
# score = 'Exemples/AccordsM-score.png'

with open('Onset_given_'+title, 'rb') as f:
# with open('Onset_given_AccordsM', 'rb') as f:
    onset_frames = pickle.load(f)

S = SignalSepare(y, sr, [y1,y2,y3], Notemin, Notemax,onset_frames, delOnsets, addOnsets, score, instrument)
S.DetectionOnsets()
# with open('Onset_given_'+title, 'wb') as g:
#      pickle.dump(S.onset_frames, g)
space = ['concordance']
#'dissonance', 'harmonicity','concordance','concordance3','concordanceTot','harmonicChange','diffConcordance','crossConcordance','crossConcordanceTot'
S.Clustering()
if params.spectrDiss_Simpl: S.SimplifySpectrum()
S.ComputeDescripteurs(space = space)
S.Affichage(space = space)






# print(S.fmin, S.fmax)
# print(np.shape(np.insert(getattr(S, 'chrom_crossConcordance')[:,1:S.n_frames-2], range(S.n_frames-2),0, axis=1)))
# print(np.shape(getattr(S, 'chrom_crossConcordance')[:,1:S.n_frames-2]))


#Nmin = int(S.sr/(S.fmax*(2**(1/BINS_PER_OCTAVE)-1)))
#Nmax = int((S.sr/(S.fmin*(2**(1/BINS_PER_OCTAVE)-1))))
