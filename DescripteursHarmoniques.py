from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

    def __init__(self, signal, sr, pistes, Notemin  = 'D3', Notemax = 'D9',  onset_frames = [], delOnsets = [], addOnsets = [], score = []):
        self.y = signal
        self.pistes = pistes
        self.sr = sr
        self.n_pistes = len(pistes)
        self.Notemin = Notemin
        self.Notemax = Notemax
        self.delOnsets = delOnsets
        self.addOnsets = addOnsets
        self.score = score
        self.n_bins_ONSETS = 0
        self.n_bins = 0
        self.N = 0
        self.fmin = 0
        self.fmax = 0
        self.n_bins = 0
        self.n_att = 0
        self.n_frames = 0
        self.times = []
        self.onset_times = []
        self.onset_frames = onset_frames
        self.Chrom_ONSETS = []
        self.ChromDB_ONSETS = []
        self.ChromSync_ONSETS = []
        self.ChromPistesSync_ONSETS = []
        self.chromSync = []
        self.chromSyncDB = []
        self.chromPistesSync = []
        self.energy = []
        self.energyPistes = []
        self.chromConc = []
        self.concordance = []
        self.chromConcTot = []
        self.concordanceTot = []
        self.tension = []
        self.dissonance = []
        self.tensionSignal = []
        self.dissonanceSignal = []
        self.chromHarmonicChange = []
        self.harmonicChange = []
        self.chromCrossConc = []
        self.crossConcordance = []
        self.chromCrossConcTot = []
        self.crossConcordanceTot = []
        self.chromDiffConcocordance = []
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
        Onset_given  = True
        if len(self.onset_frames) == 0:
            Onset_given =  False
            Diff = np.zeros((self.n_bins_ONSETS,self.N))
            Dev = np.zeros(self.N)
            for j in range(1,self.N):
                for i in range(self.n_bins_ONSETS):
                    Diff[i,j] = np.abs(self.ChromDB_ONSETS[i,j]-self.ChromDB_ONSETS[i,j-1])
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
        if params.decompo_hpss: self.Chrom, Percu = librosa.decompose.hpss(self.Chrom)
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

        #Affichage
        if params.plot_onsets:
            plt.figure(1,figsize=(13, 7))
            ax1 = plt.subplot(3, 1, 1)
            librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times)
            plt.title('CQT spectrogram')

            plt.subplot(3, 1, 2, sharex=ax1)
            if not Onset_given:
                plt.plot(self.times, Dev, label='Deviation')
                plt.plot(self.times, Seuil, color='g', label='Seuil')
                plt.vlines(self.times[Onsets], 0, Dev.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
            else:
                plt.vlines(self.times[self.onset_frames], 0, 1, color='r', alpha=0.9, linestyle='--', label='Onsets')

            plt.axis('tight')
            plt.legend(frameon=True, framealpha=0.75)

            plt.subplot(3, 1, 3, sharex=ax1)
            librosa.display.specshow(self.chromSyncDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times)
            plt.tight_layout()


        if params.plot_decompo_hpss & params.decompo_hpss:
            plt.figure(2,figsize=(13, 7))
            plt.subplot(3, 1, 1)
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.cqt(y=self.y, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins, window=WINDOW, filter_scale = FILTER_SCALE)),ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time')
            plt.title('Full cqt transform')

            plt.subplot(3, 1, 2)
            librosa.display.specshow(librosa.amplitude_to_db(self.Chrom,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time')
            plt.title('Harmonic part')

            plt.subplot(3, 1, 3)
            librosa.display.specshow(librosa.amplitude_to_db(Percu,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time')
            plt.title('Percussive part')
            plt.tight_layout()
            plt.show()


    def Clustering(self):
        """ Découpe et synchronise les pistes séparées sur les ONSETS, stoque le spectrogramme
        synchronisé en construisant self.chromPistesSync"""
# if params.decompo_hpss: self.Chrom, Percu = librosa.decompose.hpss(self.Chrom)
        #  Construction de chromPistesSync
        ChromPistes = []
        for k, voice in enumerate(self.pistes):
            if params.decompo_hpss:
                ChromPistes.append(librosa.decompose.hpss(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)))[0])
            else: ChromPistes.append(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)))

            self.chromPistesSync.append(np.zeros((self.n_bins,self.n_frames)))
            for j in range(self.n_frames):
                for i in range(self.n_bins):
                    self.chromPistesSync[k][i,j] = np.median(ChromPistes[k][i][(self.onset_frames[j]+self.n_att):(self.onset_frames[j+1]-self.n_att)])

        # Calcul de l'énergie
        self.energyPistes = np.zeros((self.n_pistes, self.n_frames))
        for t in range(self.n_frames):
            for k in range(self.n_pistes):
                self.energyPistes[k,t] = np.sum(np.multiply(self.chromPistesSync[k][:,t], self.chromPistesSync[k][:,t]))

        #  Plot
        if params.plot_pistes:
            plt.figure(3,figsize=(13, 7))
            ax1 = plt.subplot(self.n_pistes,1,1)
            librosa.display.specshow(librosa.amplitude_to_db(self.chromPistesSync[0], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times)
            for k in range(1, self.n_pistes):
                plt.subplot(self.n_pistes, 1, k+1, sharex=ax1)
                librosa.display.specshow(librosa.amplitude_to_db(self.chromPistesSync[k], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times)
            plt.tight_layout()
            plt.show()

    def Concordance(self):
        """Multiplie les spectres (cqt) des différentes pistes pour créer le spectre de concordance,
        et calcule la concordance en sommant sur les fréquences"""

        self.chromConc = np.zeros((self.n_bins,self.n_frames))
        for k in range(self.n_pistes-1):
            for l in range(k+1, self.n_pistes):
                if params.norm_conc == 'None':
                    self.chromConc = self.chromConc + np.multiply(self.chromPistesSync[k], self.chromPistesSync[l])
                if params.norm_conc == 'piste_by_piste':
                    self.chromConc = self.chromConc + np.divide(np.multiply(self.chromPistesSync[k], self.chromPistesSync[l]), np.sqrt(np.multiply(self.energyPistes[k], self.energyPistes[l])))
                elif params.norm_conc == 'energy_total':
                    self.chromConc = self.chromConc + np.divide(np.multiply(self.chromPistesSync[k], self.chromPistesSync[l]), self.energy)
        self.concordance = self.chromConc.sum(axis=0)
        self.concordance[0]=0
        self.concordance[self.n_frames-1]=0


        """elif params.norm_conc == 'energy_total':
            self.chromConc = np.zeros((self.n_bins,self.n_frames))
            for k in range(self.n_pistes-1):
                for l in range(k+1, self.n_pistes):
                    self.chromConc = self.chromConc + np.multiply(self.chromPistesSync[k], self.chromPistesSync[l])
            self.concordance = self.chromConc.sum(axis=0)
            #Normalisation
            #self.concordance = [x/y for x, y in zip(self.concordance, self.energy)]"""


    def ConcordanceTot(self):
        """Multiplie les spectres (cqt) des différentes pistes pour créer le spectre de concordance,
        et calcule la concordance en sommant sur les fréquences"""

        self.chromConcTot = np.ones((self.n_bins,self.n_frames))
        for k in range(self.n_pistes):
            self.chromConcTot = np.multiply(self.chromConcTot, self.chromPistesSync[k])
        self.concordanceTot = np.divide(self.chromConcTot.sum(axis=0),np.power(self.energy, self.n_pistes/2))

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
        if params.norm_diss: self.dissonance = np.divide(self.dissonance, self.energy)
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
                    tens = np.exp(-((int[1]-int[0])* BINS_PER_OCTAVE/(12*δ))**2)
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
                    tens = np.exp(-((int[1]-int[0])* BINS_PER_OCTAVE/(12*δ))**2)
                    self.tensionSignal = self.tensionSignal + (self.ChromSync_ONSETS[b1,:] * self.ChromSync_ONSETS[b2,:] * self.ChromSync_ONSETS[b3,:]) * tens
        self.tensionSignal = np.divide(self.tensionSignal, np.power(self.energy, 3/2))
        self.tensionSignal[0]=0
        self.tensionSignal[self.n_frames-1]=0


    def CrossConcordance(self):
        if len(self.concordance) == 0: self.Concordance()

        self.chromCrossConc = np.zeros((self.n_bins,self.n_frames-1))
        for t in range(self.n_frames-1):
            self.chromCrossConc[:,t] = np.multiply(self.chromConc[:,t],self.chromConc[:,t+1])
            if params.norm_crossConc == 'energy + conc':
                if self.concordance[t]*self.concordance[t+1]!=0:
                    self.chromCrossConc[:,t] = np.divide(self.chromCrossConc[:,t], self.concordance[t]*self.concordance[t+1])
        self.crossConcordance = self.chromCrossConc.sum(axis=0)
        self.crossConcordance[0]=0
        self.crossConcordance[self.n_frames-2]=0


    def CrossConcordanceTot(self):
        if len(self.concordanceTot) == 0: self.ConcordanceTot()

        self.chromCrossConcTot = np.zeros((self.n_bins,self.n_frames-1))
        for t in range(self.n_frames-1):
            self.chromCrossConcTot[:,t] = np.multiply(self.chromConcTot[:,t],self.chromConcTot[:,t+1])
            if params.norm_crossConcTot == 'energy + concTot':
                if self.concordanceTot[t]*self.concordanceTot[t+1]!=0:
                    self.chromCrossConcTot[:,t] = np.divide(self.chromCrossConcTot[:,t], self.concordanceTot[t]*self.concordanceTot[t+1])
        self.crossConcordanceTot = self.chromCrossConcTot.sum(axis=0)
        self.crossConcordanceTot[0]=0
        self.crossConcordanceTot[self.n_frames-2]=0


    def HarmonicChange(self):
        self.chromHarmonicChange = np.zeros((self.n_bins,self.n_frames-1))
        for t in range(self.n_frames-1):
            # Par défaut, params.type_harmChange == 'relative'


            if params.norm_harmChange == 'None':
                self.chromHarmonicChange[:,t] = self.chromSync[:,t+1] - self.chromSync[:,t]

            elif params.norm_harmChange == 'frame_by_frame':
                self.chromHarmonicChange[:,t] = np.subtract([x/np.sqrt(self.energy[t+1]) for x in self.chromSync[:,t+1]], [y/np.sqrt(self.energy[t]) for y in self.chromSync[:,t]])

            elif params.norm_harmChange == 'general':
                self.chromHarmonicChange[:,t] = np.divide(self.chromSync[:,t+1] - self.chromSync[:,t], (self.energy[t+1]*self.energy[t])**(1/4))

            if params.type_harmChange == 'absolute': self.chromHarmonicChange[:,t] = np.abs(self.chromHarmonicChange[:,t])

        self.harmonicChange = self.chromHarmonicChange.sum(axis=0)
        self.harmonicChange[0]=0
        self.harmonicChange[self.n_frames-2]=0


    def DiffConcordance(self):
        self.chromDiffConc = np.zeros((self.n_bins,self.n_frames-1))
        for k in range(self.n_pistes):
            for l in range(self.n_pistes):
                if k != l:
                    for t in range(self.n_frames-1):
                        if params.norm_diffConc == 'piste_by_piste':
                            self.chromDiffConc[:,t] = self.chromDiffConc[:,t] + np.divide(np.multiply(self.chromPistesSync[k][:,t], self.chromPistesSync[l][:,t+1]), np.sqrt(self.energyPistes[k][t] * self.energyPistes[l][t+1]))
                        elif params.norm_diffConc == 'frame-by-frame':
                            self.chromDiffConc[:,t] = self.chromDiffConc[:,t] + np.divide(np.multiply(self.chromPistesSync[k][:,t], self.chromPistesSync[l][:,t+1]), np.sqrt(self.energy[t] * self.energy[t+1]))


        self.diffConcordance = self.chromDiffConc.sum(axis=0)
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

        # Séparation contribution au virtual pitch / le reste





    def ComputeDescripteurs(self, space = ['concordance','concordanceTot']):
        """Calcule les descripteurs indiqués dans 'space', puis les affiche"""

        dim = len(space)
        if 'concordance' in space: self.Concordance()
        if 'concordanceTot' in space: self.ConcordanceTot()
        if 'tension' in space: self.Tension()
        if 'dissonance' in space: self.Dissonance()
        if 'tensionSignal' in space: self.TensionSignal()
        if 'dissonanceSignal' in space: self.DissonanceSignal()
        if 'harmonicity' in space: self.Harmonicity()
        if 'crossConcordance' in space: self.CrossConcordance()
        if 'crossConcordanceTot' in space: self.CrossConcordanceTot()
        if 'harmonicChange' in space: self.HarmonicChange()
        if 'diffConcordance' in space: self.DiffConcordance()
        #print(np.shape(np.asarray(self.chromConc)))

        #Plot les spectrogrammes
        if params.plot_chromDescr:
            plt.figure(4,figsize=(13, 7))
            ax1 = plt.subplot(3,1,1)
            librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times)
            plt.title('CQT spectrogram')

            plt.subplot(3, 1, 2, sharex=ax1)
            librosa.display.specshow(librosa.amplitude_to_db(self.chromConc, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times)
            plt.title('Spectre de concordance')

            plt.subplot(3, 1, 3, sharex=ax1)
            librosa.display.specshow(librosa.amplitude_to_db(self.chromConcTot, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times)
            plt.title('Spectre de concordance Totale')
            plt.tight_layout()
            plt.show()

        #Plot les descripteurs harmoniques
        if params.plot_descr:

            plt.figure(5,figsize=(13, 7))
            s = int(params.plot_score)

            # Partition
            if params.plot_score & (len(self.score)!=0):
                #plt.subplot(dim+1+s,1,s)
                img=mpimg.imread(self.score)
                score = plt.subplot(dim+1+s,1,1)
                plt.axis('off')
                score.imshow(img)


            ax1 = plt.subplot(dim+1+s,1,1+s)
            librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times)
            plt.title(title)

            for k, descr in enumerate(space):
                plt.subplot(dim+1+s, 1, k+2+s, sharex=ax1)
                plt.vlines(self.onset_times[1:self.n_frames], min(getattr(self, descr)), max(getattr(self, descr)), color='k', alpha=0.9, linestyle='--')
                if not all(x>=0 for x in getattr(self, descr)):
                    plt.hlines(0,self.onset_times[0], self.onset_times[self.n_frames], alpha=0.5, linestyle = ':')

                # Descripteurs statiques
                if len(getattr(self, descr)) == self.n_frames:
                    plt.hlines(getattr(self, descr)[1:(self.n_frames-1)], self.onset_times[1:(self.n_frames-1)], self.onset_times[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][k], label=descr)
                # Descripteurs dynamiques
                elif len(getattr(self, descr)) == (self.n_frames-1):
                    plt.plot(self.onset_times[2:(self.n_frames-1)], getattr(self, descr)[1:(self.n_frames-2)],['b','r','g','c','m','y','b','r','g'][k]+'o', label=descr)
                    plt.hlines(getattr(self, descr)[1:(self.n_frames-2)], [t-0.25 for t in self.onset_times[2:(self.n_frames-1)]], [t+0.25 for t in self.onset_times[2:(self.n_frames-1)]], color=['b','r','g','c','m','y','b','r','g'][k], alpha=0.9, linestyle=':'  )
                plt.legend(frameon=True, framealpha=0.75)
            plt.tight_layout()
            plt.show()

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
                    ax.set_xlim(0, max(xdescr)*6/5)
                    ax.set_ylim(0, max(ydescr)*6/5)
                    if play : a.play()
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



title = 'SuiteAccordsPiano'
#Palestrina, PalestrinaM, SuiteAccords, UnSeulAccord, AccordsParalleles, AccordRepete, 'SuiteAccordsOrgue', 'SuiteAccordsPiano'
y, sr = librosa.load('Exemples/'+title+'.wav')
y1, sr = librosa.load('Exemples/'+title+'-Basse.wav')
y2, sr = librosa.load('Exemples/'+title+'-Alto.wav')
y3, sr = librosa.load('Exemples/'+title+'-Soprano.wav')
delOnsets = []
addOnsets = []
if params.SemiManual:
    delOnsets = getattr(params, 'delOnsets'+'_'+title)
    addOnsets = getattr(params, 'addOnsets'+'_'+title)
    α = getattr(params, 'paramsDetOnsets'+'_'+title)[0]
    β = getattr(params, 'paramsDetOnsets'+'_'+title)[1]
    H = getattr(params, 'paramsDetOnsets'+'_'+title)[2]
    T = getattr(params, 'paramsDetOnsets'+'_'+title)[3]
    T_att = getattr(params, 'paramsDetOnsets'+'_'+title)[4]



Notemin = 'D3' #'SuiteAccordsOrgue': A2
Notemax = 'D9'
score = 'Exemples/Score_SuiteAccords.png'
with open('Onset_given_SuiteAccordsPiano', 'rb') as f:
    onset_frames = pickle.load(f)

S = SignalSepare(y, sr, [y1,y2,y3], Notemin, Notemax, onset_frames, delOnsets, addOnsets, score)
S.DetectionOnsets()
# with open('Onset_given_SuiteAccordsPiano', 'wb') as f:
#      pickle.dump(S.onset_frames, f)
S.Clustering()
S.ComputeDescripteurs(space = ['concordance', 'concordanceTot'])


#Nmin = int(S.sr/(S.fmax*(2**(1/BINS_PER_OCTAVE)-1)))
#Nmax = int((S.sr/(S.fmin*(2**(1/BINS_PER_OCTAVE)-1))))
