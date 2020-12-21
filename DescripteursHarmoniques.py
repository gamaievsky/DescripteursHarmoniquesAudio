from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter, attrgetter, truediv
import statistics as stat
from scipy import signal
from scipy.optimize import minimize
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
np.seterr(divide='ignore', invalid='ignore')


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

ϵ = sys.float_info.epsilon




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
        self.N_sample = []
        self.Dev = []
        self.Seuil = []
        self.times = []
        self.Onset_given = True
        self.onset_times = []
        self.onset_times_graph = []
        self.onset_frames = onset_frames
        self.Percu = []
        self.Chrom_ONSETS = []
        self.ChromDB_ONSETS = []
        self.ChromSync_ONSETS = []
        self.ChromPistesSync_ONSETS = []
        self.ChromDB_reloc = []
        self.Chrom = []
        self.chromSync = []
        self.chromSyncDB = []
        self.chromPistesSync = []
        self.chromSyncSimpl = []
        self.chromPistesSyncSimpl = []
        self.ChromNoHpss = []
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
        self.roughness = []
        self.chrom_harmonicity = []
        self.liste_partials = []
        self.tensionSignal = []
        self.chrom_roughness = []
        self.roughnessSignal = []
        self.chrom_harmonicChange = []
        self.harmonicChange = []
        self.chrom_crossConcordance = []
        self.crossConcordance = []
        self.chrom_crossConcordanceTot = []
        self.crossConcordanceTot = []
        self.chrom_diffConcordance = []
        self.diffRoughness = []
        self.chrom_diffRoughness = []
        self.diffConcordance = []
        self.harmonicity = []
        self.virtualPitch = []
        self.context = []
        self.contextSimpl = []
        self.energyContext = []
        self.chrom_harmonicNovelty = []
        self.harmonicNovelty = []
        self.harmonicityContext = []
        self.virtualPitchContext = []
        self.roughnessContext = []
        self.chrom_roughnessContext = []
        self.diffConcordanceContext = []
        self.chrom_diffConcordanceContext = []
        self.diffRoughnessContext = []
        self.chrom_diffRoughnessContext = []






    def DetectionOnsets(self):
        self.fmin = librosa.note_to_hz(self.Notemin)
        self.fmax = librosa.note_to_hz(self.Notemax)
        #Nmin = int((sr/(fmax*(2**(1/BINS_PER_OCTAVE)-1))))
        #Nmax = int((sr/(fmin*(2**(1/BINS_PER_OCTAVE)-1))))
        self.n_bins_ONSETS = int((librosa.note_to_midi(self.Notemax) - librosa.note_to_midi(self.Notemin))*BINS_PER_OCTAVE_ONSETS/12)
        self.Chrom_ONSETS = np.abs(librosa.cqt(y=self.y, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE_ONSETS, n_bins=self.n_bins_ONSETS, window=WINDOW))
        self.ChromDB_ONSETS = librosa.amplitude_to_db(self.Chrom_ONSETS, ref=np.max)
        self.N = len(self.ChromDB_ONSETS[0])
        self.times = librosa.frames_to_time(np.arange(self.N), sr=self.sr, hop_length=STEP)

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
            if isinstance(self.delOnsets, str): Onsets = []
            else:
                self.delOnsets.sort(reverse = True)
                for o in self.delOnsets:
                    Onsets.pop(o-1)


             #Ajout manuel des onsets
            for t in self.addOnsets:
                Onsets.append(librosa.time_to_frames(t, sr=self.sr, hop_length=STEP))
                Onsets.sort()
            self.onset_frames = librosa.util.fix_frames(Onsets, x_min=0, x_max=self.ChromDB_ONSETS.shape[1]-1)

        self.onset_frames = librosa.util.fix_frames(self.onset_frames, x_min=0, x_max=self.ChromDB_ONSETS.shape[1]-1)
        self.onset_times = librosa.frames_to_time(self.onset_frames, sr=self.sr, hop_length = STEP)
        self.n_frames = len(self.onset_frames)-1
        self.n_notes = np.ones(self.n_frames)

        # TRANSFORMÉE avec la précision due pour l'analyse
        self.n_bins = int((librosa.note_to_midi(self.Notemax) - librosa.note_to_midi(self.Notemin))*BINS_PER_OCTAVE/12)
        self.Chrom = np.abs(librosa.cqt(y=self.y, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins, window=WINDOW, filter_scale = FILTER_SCALE))
        # self.Chrom[np.isnan(self.Chrom)] = 0


        # Relocalisation
        if params.spectral_reloc:
            freq_analyse = [self.fmin*2**(k/BINS_PER_OCTAVE) for k in range(self.n_bins)]
            N = [round(self.sr * params.FILTER_SCALE/(f*(2**(1/BINS_PER_OCTAVE)-1))) for f in freq_analyse]
            self.N_sample = [round(n/STEP) for n in N]
            Chrom_copy = np.copy(self.Chrom)
            for k in range(self.n_bins):
                for n in reversed(range(self.N)):
                    if n <= self.N_sample[k]: self.Chrom[k,n] = Chrom_copy[k,n]
                    else: self.Chrom[k,n] = Chrom_copy[k,n-int(self.N_sample[k]/2)]

        # Décomposition partie harmonique / partie percussive
        if params.decompo_hpss:
            self.ChromNoHpss = np.copy(self.Chrom)
            self.Chrom = librosa.decompose.hpss(self.Chrom, margin=params.margin)[0]

        self.ChromDB = librosa.amplitude_to_db(self.Chrom, ref=np.max)


        #Synchronisation sur les onsets, en enlevant le début et la fin des longues frames
        self.chromSync = np.zeros((self.n_bins,self.n_frames))
        self.n_att = int(librosa.time_to_frames(T_att, sr=self.sr, hop_length = STEP))
        # for j in range(self.n_frames):
        #     if j==0:
        #         for i in range(self.n_bins):
        #             self.chromSync[i,j] = np.median(self.Chrom[i][self.onset_frames[j]:self.onset_frames[j+1]])
        #     else:
        #         for i in range(self.n_bins):
        #             self.chromSync[i,j] = np.median(self.Chrom[i][(self.onset_frames[j]+self.n_att):(self.onset_frames[j+1])])
        Δmin = 0.1 # en secondes
        for i in range(self.n_bins):
            f = self.fmin*2**(i/BINS_PER_OCTAVE)
            T_ret = 1.5 / (f * (2**(1.0/(12*4)) - 1))
            for j in range(self.n_frames):
                if j==0: self.chromSync[i,j] = np.median(self.Chrom[i][self.onset_frames[j]:self.onset_frames[j+1]])
                else:
                    if T_ret < (self.onset_times[j+1] - self.onset_times[j+1]) - Δmin:
                        self.chromSync[i,j] = np.median(self.Chrom[i][(self.onset_frames[j]+int(librosa.time_to_frames(T_ret, sr=self.sr, hop_length = STEP))):(self.onset_frames[j+1])])
                    else:
                        self.chromSync[i,j] = np.median(self.Chrom[i][(self.onset_frames[j+1]-int(librosa.time_to_frames(Δmin, sr=self.sr, hop_length = STEP))):(self.onset_frames[j+1])])




        self.chromSync[np.isnan(self.chromSync)] = 0
        self.chromSync[:,0] = np.zeros(self.n_bins)
        self.chromSync[:,-1] = np.zeros(self.n_bins)
        self.chromSyncDB = librosa.amplitude_to_db(self.chromSync, ref=np.max)


        #Calcul de l'énergie
        for t in range(self.n_frames):
            self.energy.append(LA.norm(self.chromSync[:,t])**2)


    def Clustering(self):
        """ Découpe et synchronise les pistes séparées sur les ONSETS, stoque le spectrogramme
        synchronisé en construisant self.chromPistesSync"""

        if len(self.pistes) != 0:
            #  Construction de chromPistesSync
            ChromPistes = []
            for k, voice in enumerate(self.pistes):
                if params.decompo_hpss:
                    ChromPistes.append(np.nan_to_num(librosa.decompose.hpss(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)), margin=params.margin)[0],False))
                else: ChromPistes.append(np.nan_to_num(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)),False))


            for k, voice in enumerate(self.pistes):
                # if params.spectral_reloc:
                #     ChromPistesCopy = np.copy(ChromPistes)
                #     for f in range(self.n_bins):
                #         for n in reversed(range(self.N)):
                #             if n <= self.N_sample[f]: ChromPistes[k][f,n] = ChromPistesCopy[k][f,n]
                #             else: ChromPistes[k][f,n] = ChromPistesCopy[k][f,n-int(self.N_sample[f]/2)]

                self.chromPistesSync.append(np.zeros((self.n_bins,self.n_frames)))
                for j in range(self.n_frames):
                    if j==0:
                        for i in range(self.n_bins):
                            self.chromPistesSync[k][i,j] = np.median(ChromPistes[k][i][self.onset_frames[j]:self.onset_frames[j+1]])
                    else:
                        for i in range(self.n_bins):
                            self.chromPistesSync[k][i,j] = np.median(ChromPistes[k][i][(self.onset_frames[j]+self.n_att):(self.onset_frames[j+1]-self.n_att)])

            # Calcul de l'énergie des pistes
            self.energyPistes = np.zeros((self.n_pistes, self.n_frames))
            for t in range(self.n_frames):
                for k in range(self.n_pistes):
                    self.energyPistes[k,t] = np.sum(np.multiply(self.chromPistesSync[k][:,t], self.chromPistesSync[k][:,t]))

            # Calcul de la matrice d'activation (pour savoir quelles voix contiennent des notes à quel moment) + mise à zéro des pistes sans note
            # Par défaut, tout est à 1, ie dans chaque frame chaque piste joue une note
            self.activation = np.ones((self.n_pistes, self.n_frames))
            if title in params.list_calcul_nombre_notes:
                max_energy = np.amax(self.energyPistes, axis = 1)
                for k in range(self.n_pistes):
                    for t in range(self.n_frames):
                        if (self.energyPistes[k,t] < params.seuil_activation * max_energy[k]):
                            self.activation[k,t] = 0
                            self.chromPistesSync[k][:,t] = 0
                self.activation[:,0] = 0
                self.activation[:,self.n_frames-1] = 0
            elif title in params.list_3_voix:
                self.activation[-1] = np.zeros(self.n_frames)


            # Calcul du nombre de notes
            self.n_notes = np.sum(self.activation, axis=0)
            self.n_notes[0] = 0
            self.n_notes[-1] = 0


    def Context(self, type = params.memory_type, size = params.memory_size - 1, decr = params.memory_decr_ponderation):

        #Construction du context harmonique
        self.context = np.zeros((self.n_bins,self.n_frames))
        self.context[:,0] = self.chromSync[:,0]

        # Memory = "full"
        if isinstance(size,str):
            if type == 'max':
                for t in range(1,self.n_frames):
                    self.context[:,t] = np.fmax(self.context[:,t-1],self.chromSync[:,t])
            elif type == 'mean':
                #Construction du vecteur de pondération
                weights = [(1/i**decr) for i in range(1,self.n_frames+2)]
                #Moyennage
                for t in range(1,self.n_frames):
                    self.context[:,t] = np.average(self.chromSync[:,:(t+1)], axis=1, weights=[weights[t-i] for i in range(t+1)])

        # Memory = int
        elif isinstance(size,int):
            if type == 'max':
                for t in range(1,size+1):
                    self.context[:,t] = np.fmax(self.chromSync[:,t], self.context[:,t-1])
                for t in range(size+1,self.n_frames):
                    self.context[:,t] = np.amax(self.chromSync[:,(t-size):(t+1)], axis = 1)
            elif type == 'mean':
                #Construction du vecteur de pondération
                weights = [(1/i**decr) for i in range(1,size+2)]
                #Moyennage
                for t in range(1,size+1):
                    self.context[:,t] = np.average(self.chromSync[:,1:(t+1)], axis=1, weights=[weights[t-i] for i in range(1,t+1)])
                for t in range(size+1,self.n_frames):
                    self.context[:,t] = np.average(self.chromSync[:,(t-size):(t+1)], axis=1, weights=[weights[size-i] for i in range(size+1)])
        #Calcul de l'énergie du contexte
        self.energyContext = []
        for t in range(self.n_frames):
            self.energyContext.append(LA.norm(self.context[:,t])**2)

    def SimplifySpectrum(self):
        self.chromSyncSimpl = np.zeros(self.chromSync.shape)
        self.contextSimpl = np.zeros(self.context.shape)
        self.chromPistesSyncSimpl= np.copy(self.chromPistesSync)
        δ = params.δ
        if distribution == 'themeAcc' or distribution == 'voix':
            for t in range(self.n_frames):
                for i in range(10, self.n_bins - 10):
                    for p in range(len(self.pistes)):
                        if self.chromPistesSync[p][i,t] < np.max(self.chromPistesSync[p][i-δ:i+δ+1,t]): self.chromPistesSyncSimpl[p][i,t] = 0
                        else: self.chromPistesSyncSimpl[p][i,t] = np.sum(self.chromPistesSync[p][i-δ:i+δ+1,t])
                    # Global
                    if self.chromSync[i,t] < np.max(self.chromSync[i-δ:i+δ+1,t]): self.chromSyncSimpl[i,t] = 0
                    else: self.chromSyncSimpl[i,t] = np.sum(self.chromSync[i-δ:i+δ+1,t])
                    # Contexte
                    if self.context[i,t] < np.max(self.context[i-δ:i+δ+1,t]): self.contextSimpl[i,t] = 0
                    else: self.contextSimpl[i,t] = np.sum(self.context[i-δ:i+δ+1,t])

            # for p in range(len(self.pistes)):
            #     self.chromSyncSimpl += self.chromPistesSyncSimpl[p]
        elif distribution == 'record':
            for t in range(self.n_frames):
                for i in range(10, self.n_bins - 10):
                    # Global
                    if self.chromSync[i,t] < np.max(self.chromSync[i-δ:i+δ+1,t]): self.chromSyncSimpl[i,t] = 0
                    else: self.chromSyncSimpl[i,t] = np.sum(self.chromSync[i-δ:i+δ+1,t])
                    # Contexte
                    if self.context[i,t] < np.max(self.context[i-δ:i+δ+1,t]): self.contextSimpl[i,t] = 0
                    else: self.contextSimpl[i,t] = np.sum(self.context[i-δ:i+δ+1,t])

        # Liste des partiels de self.chromSyncSimpl
        self.liste_partials = []
        for t in range(self.n_frames):
            self.liste_partials.append([])
            for k in range(self.n_bins):
                if self.chromSyncSimpl[k,t] > 0: self.liste_partials[t].append(k)


    def Concordance(self):
        """Multiplie les spectres (cqt) des différentes pistes pour créer le spectre de concordance,
        et calcule la concordance en sommant sur les fréquences"""

        self.chrom_concordance = np.zeros((self.n_bins,self.n_frames))
        for k in range(self.n_pistes-1):
            for l in range(k+1, self.n_pistes):
                self.chrom_concordance += np.multiply(self.chromPistesSync[k], self.chromPistesSync[l])

        # Normalisation par l'énergie et par le nombre de notes
        for t in range(self.n_frames):
            if self.n_notes[t] >= 2:
                self.chrom_concordance[:,t] *= (self.n_notes[t]**(2*params.norm_conc)/(self.n_notes[t]*(self.n_notes[t]-1)/2)) / (self.energy[t]**params.norm_conc)

        self.chrom_concordance[:,0] = 0
        self.chrom_concordance[:,self.n_frames-1] = 0
        self.concordance = self.chrom_concordance.sum(axis=0)
        # self.concordance[0]=0
        # self.concordance[self.n_frames-1]=0


    def ConcordanceTot(self):
        """Multiplie les spectres (cqt) des différentes pistes pour créer le spectre de concordance,
        et calcule la concordance en sommant sur les fréquences"""

        self.chrom_concordanceTot = np.ones((self.n_bins,self.n_frames))
        for t in range(self.n_frames):
            for k in range(self.n_pistes):
                if self.activation[k,t]:
                    self.chrom_concordanceTot[:,t] = np.multiply(self.chrom_concordanceTot[:,t], self.chromPistesSync[k][:,t])
            if self.n_notes[t]>=1:
                self.chrom_concordanceTot[:,t] = np.divide((self.n_notes[t]**(self.n_notes[t]*params.norm_concTot)) * self.chrom_concordanceTot[:,t], LA.norm(self.chromSync[:,t], self.n_notes[t])**(self.n_notes[t]*params.norm_concTot))
            self.concordanceTot.append(self.chrom_concordanceTot[:,t].sum(axis=0))#**(1./self.n_notes[t]))


        self.chrom_concordanceTot[:,0] = 0
        self.chrom_concordanceTot[:,self.n_frames-1] = 0
        self.concordanceTot[0]=0
        self.concordanceTot[self.n_frames-1]=0


    def Concordance3(self):
        self.chrom_concordance3 = np.zeros((self.n_bins,self.n_frames))
        for k in range(self.n_pistes-2):
            for l in range(k+1, self.n_pistes-1):
                for m in range(l+1, self.n_pistes):
                    self.chrom_concordance3 += np.multiply(np.multiply(self.chromPistesSync[k], self.chromPistesSync[l]), self.chromPistesSync[m])

        # Normalisation par la norme 3 et le nombre de notes
        for t in range(self.n_frames):
            if self.n_notes[t] >= 3:
                self.chrom_concordance3[:,t] *= (self.n_notes[t]**(3*params.norm_conc3)/(self.n_notes[t]*(self.n_notes[t]-1)*(self.n_notes[t]-2)/6)) / LA.norm(self.chromSync[:,t],ord=3)**(3*params.norm_conc3)

        self.chrom_concordance3[:,0] = 0
        self.chrom_concordance3[:,self.n_frames-1] = 0
        self.concordance3 = self.chrom_concordance3.sum(axis=0)
        # self.concordance3[0]=0
        # self.concordance3[self.n_frames-1]=0


    def Roughness(self):
        #fonction qui convertit les amplitudes en sonies
        # def Sones(Ampl):
        #     P = Ampl/np.sqrt(2)
        #     SPL = 20*np.log10(P/params.P_ref)
        #     return ((1/16)*np.power(2,SPL/10))
        self.chrom_roughness = np.zeros((self.n_bins,self.n_frames))
        self.roughness = np.zeros(self.n_frames)

        for b1 in range(self.n_bins-1):
            for b2 in range(b1+1,self.n_bins):
                # Modèle de Sethares
                f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                freq = [f1, f2]
                freq.sort()
                if params.mod_rough == 'sethares + KK':
                    s = (1/2.27)*(np.log(params.β2/params.β1)/(params.β2-params.β1))/(freq[0]**(0.477))
                elif params.mod_rough == 'sethares':
                    s = 0.24/(0.021*freq[0] + 19)
                rug = np.exp(-params.β1*s*(freq[1]-freq[0]))-np.exp(-params.β2*s*(freq[1]-freq[0]))

                if not params.type_rug_signal:
                    for p1 in range(self.n_pistes-1):
                        for p2 in range(p1+1, self.n_pistes):
                            if params.rug_simpl:
                                self.chrom_roughness[b1] += (self.chromPistesSyncSimpl[p1][b1] * self.chromPistesSyncSimpl[p2][b2] + self.chromPistesSyncSimpl[p1][b2] * self.chromPistesSyncSimpl[p2][b1]) * rug/2
                                self.chrom_roughness[b2] += (self.chromPistesSyncSimpl[p1][b1] * self.chromPistesSyncSimpl[p2][b2] + self.chromPistesSyncSimpl[p1][b2] * self.chromPistesSyncSimpl[p2][b1]) * rug/2
                            else:
                                self.chrom_roughness[b1] += (self.chromPistesSync[p1][b1] * self.chromPistesSync[p2][b2] + self.chromPistesSync[p1][b2] * self.chromPistesSync[p2][b1]) * rug/2
                                self.chrom_roughness[b2] += (self.chromPistesSync[p1][b1] * self.chromPistesSync[p2][b2] + self.chromPistesSync[p1][b2] * self.chromPistesSync[p2][b1]) * rug/2
                else:
                    if params.rug_simpl:
                        self.chrom_roughness[b1] += (self.chromSyncSimpl[b1] * self.chromSyncSimpl[b2]) * rug/2
                        self.chrom_roughness[b2] += (self.chromSyncSimpl[b1] * self.chromSyncSimpl[b2]) * rug/2
                    else:
                        self.chrom_roughness[b1] += (self.chromSync[b1] * self.chromSync[b2]) * rug/2
                        self.chrom_roughness[b2] += (self.chromSync[b1] * self.chromSync[b2]) * rug/2


        # Normalisation par l'énergie et par le nombre de n_notes
        for t in range(self.n_frames):
            if not params.type_rug_signal:
                if self.n_notes[t] >= 2:
                    self.chrom_roughness[:,t] *= (self.n_notes[t]**(2*params.norm_rug) / (self.n_notes[t]*(self.n_notes[t]-1)/2.0)) / (self.energy[t]**params.norm_rug)
            else:
                self.chrom_roughness[:,t] /= self.energy[t]**params.norm_rug
        self.chrom_roughness[:,0] = 0
        self.roughness = self.chrom_roughness.sum(axis=0)
        self.roughness[0]=0
        self.roughness[self.n_frames-1]=0


    def RoughnessSignal(self):
        self.roughnessSignal = np.zeros(self.n_frames)
        for b1 in range(self.n_bins-1):
            for b2 in range(b1+1,self.n_bins):
                # Modèle de Sethares
                f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                freq = [f1, f2]
                freq.sort()
                s = 0.44*(np.log(params.β2/params.β1)/(params.β2-params.β1))*(freq[1]-freq[0])/(freq[0]**(0.477))
                rug = np.exp(-params.β1*s)-np.exp(-params.β2*s)

                self.roughnessSignal = self.roughnessSignal + (self.chromSync[b1,:] * self.chromSync[b2,:]) * rug
        if params.norm_rug: self.roughnessSignal = np.divide(self.roughnessSignal, np.power(self.energy,1.0))
        self.roughnessSignal[0]=0
        self.roughnessSignal[self.n_frames-1]=0


    def Tension(self):

        self.tension = np.zeros(self.n_frames)
        # liste des partiels
        set_partials = set()
        for t in range(self.n_frames):
            set_partials = set_partials.union(set(self.liste_partials[t]))
        liste_partials_full = list(set_partials)
        liste_partials_full.sort()


        # Calcul de la tension
        long = len(liste_partials_full)
        for i1 in range(long-2) :
            for i2 in range(i1+1,long-1):
                for i3 in range(i2+1,long):
                    int1, int2 = liste_partials_full[i2] - liste_partials_full[i1], liste_partials_full[i3] - liste_partials_full[i2]
                    tens = np.exp(-((int2-int1) / (0.6*BINS_PER_OCTAVE/12.0))**2)
                    for t in range(self.n_frames):
                        self.tension[t] += self.chromSyncSimpl[liste_partials_full[i1],t] * self.chromSyncSimpl[liste_partials_full[i2],t] * self.chromSyncSimpl[liste_partials_full[i3],t] * tens

        # Normalisation par l'énergie et par le nombre de n_notes
        for t in range(self.n_frames):
            self.tension[t] /= self.energy[t]**(params.norm_tension*3/2.0)
            # if self.n_notes[t] >= 3:
                # self.tension[t] *= (self.n_notes[t]**(3*params.norm_tension) / (self.n_notes[t]*(self.n_notes[t]-1)*(self.n_notes[t]-2)/6.0)) / (self.energy[t]**(params.norm_tension*3/2.0))

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
            if params.norm_crossConc:
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
            if params.norm_crossConcTot:
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
                self.chrom_harmonicChange[:,t] = self.chromSync[:,t+1] - self.context[:,t]

            elif params.norm_harmChange == 'frame_by_frame':
                self.chrom_harmonicChange[:,t] = self.chromSync[:,t+1]/np.sqrt(self.energy[t+1]) - self.context[:,t]/np.sqrt(self.energyContext[t])

            elif params.norm_harmChange == 'general':
                self.chrom_harmonicChange[:,t] = (self.chromSync[:,t+1] - self.context[:,t]) / (self.energy[t+1]*self.energyContext[t])**(1.0/4)

            if params.type_harmChange == 'absolute': self.chrom_harmonicChange[:,t] = np.abs(self.chrom_harmonicChange[:,t])

        self.harmonicChange = np.sum(np.power(self.chrom_harmonicChange,1), axis=0)
        self.harmonicChange[0]=0
        self.harmonicChange[-1]=0


    def HarmonicNovelty(self):
        if len(self.context) == 0: self.context()

        # Construction du spectre des Nouveautés harmoniques
        self.chrom_harmonicNovelty = np.zeros((self.n_bins,self.n_frames))
        self.chrom_harmonicNovelty[:,1] = self.chromSync[:,1]
        for t in range(2,self.n_frames):
            if params.norm_Novelty == 'frame_by_frame':
                self.chrom_harmonicNovelty[:,t] = self.chromSync[:,t]/np.sqrt(self.energy[t]) - self.context[:,t-1]/np.sqrt(self.energyContext[t-1])
            elif params.norm_Novelty == 'general':
                self.chrom_harmonicNovelty[:,t] = np.divide(self.chromSync[:,t] - self.context[:,t-1], (self.energy[t]*self.energyContext[t-1])**(1/4))
            elif params.norm_Novelty == 'None':
                self.chrom_harmonicNovelty[:,t] = self.chromSync[:,t] - self.context[:,t-1]
            for i in range(self.n_bins):
                if self.chrom_harmonicNovelty[:,t][i]<0: self.chrom_harmonicNovelty[:,t][i] = 0

        # Construction des Nouveautés harmoniques
        self.harmonicNovelty = np.exp(self.chrom_harmonicNovelty.sum(axis=0))
        if params.type_Novelty == 'dyn':
            self.harmonicNovelty = self.harmonicNovelty[1:]
        self.harmonicNovelty[0]=0
        self.harmonicNovelty[-1]=0


    def DiffConcordance(self):
        self.chrom_diffConcordance = np.zeros((self.n_bins,self.n_frames-1))
        if not params.theme_diffConc:
            for t in range(self.n_frames-1):
                self.chrom_diffConcordance[:,t] = np.multiply(self.chromSync[:,t], self.chromSync[:,t+1])
                self.chrom_diffConcordance[:,t] /= np.sqrt(self.energy[t] * self.energy[t+1]) ** params.norm_diffConc
        else :
            for t in range(self.n_frames-1):
                if self.activation[0,t+1]:
                    self.chrom_diffConcordance[:,t] = np.multiply(self.chromSync[:,t], self.chromPistesSync[0][:,t+1]) / np.sqrt(self.energy[t] * self.energyPistes[0][t+1])** params.norm_diffConc

        self.diffConcordance = self.chrom_diffConcordance.sum(axis=0)
        self.diffConcordance[0]=0
        self.diffConcordance[self.n_frames-2]=0


    def Harmonicity(self):
        # Construction du spectre harmonique
        dec = BINS_PER_OCTAVE/6 # décalage d'un ton pour tenir compte de l'épaisseur des gaussiennes
        epaiss = int(np.rint(BINS_PER_OCTAVE/(2*params.σ)))
        print(epaiss)
        SpecHarm = np.zeros(2*int(dec) + int(np.rint(BINS_PER_OCTAVE * np.log2(params.κ))))
        for k in range(params.κ):
            pic =  int(dec + np.rint(BINS_PER_OCTAVE * np.log2(k+1)))
            for i in range(-epaiss, epaiss+1):
                SpecHarm[pic + i] = 1/(k+1)**params.decr
        len_corr = self.n_bins + len(SpecHarm) - 1

        # Correlation avec le spectre réel
        self.chrom_harmonicity = np.zeros((len_corr,self.n_frames))
        self.harmonicity = []
        for t in range(self.n_frames):
            self.chrom_harmonicity[:,t] = np.correlate(np.power(self.chromSync[:,t],params.norm_harmonicity), SpecHarm,"full") / self.energy[t]**(params.norm_harmonicity * params.norm_harm/2)
            self.harmonicity.append(np.exp(max(self.chrom_harmonicity[:,t])))

            # Virtual Pitch
            self.virtualPitch.append(np.argmax(self.chrom_harmonicity[:,t]))

        virtualNotes = librosa.hz_to_note([self.fmin * (2**((i-len(SpecHarm)+dec+1)/BINS_PER_OCTAVE)) for i in self.virtualPitch] , cents = False)
        global f_corr_min
        f_corr_min =  self.fmin * (2**((-len(SpecHarm)+dec+1)/BINS_PER_OCTAVE))
        print(virtualNotes[1:self.n_frames-1])
        self.chrom_harmonicity[:,0] = 0
        self.chrom_harmonicity[:,self.n_frames-1] = 0
        self.harmonicity[0]=0
        self.harmonicity[self.n_frames-1]=0


    def HarmonicityContext(self):
        # Construction du spectre harmonique
        dec = BINS_PER_OCTAVE/6 # décalage d'un ton pour tenir compte de l'épaisseur des gaussiennes
        epaiss = int(np.rint(BINS_PER_OCTAVE/(2*params.σ)))
        SpecHarm = np.zeros(2*int(dec) + int(np.rint(BINS_PER_OCTAVE * np.log2(params.κ))))
        for k in range(params.κ):
            pic =  int(dec + np.rint(BINS_PER_OCTAVE * np.log2(k+1)))
            for i in range(-epaiss, epaiss+1):
                SpecHarm[pic + i] = 1/(k+1)**params.decr

        # Correlation avec le spectre réel
        self.harmonicityContext = []
        for t in range(self.n_frames):
            self.harmonicityContext.append(max(np.correlate(np.power(self.context[:,t],params.norm_harmonicity), SpecHarm,"full")) / LA.norm(self.context[:,t], ord = params.norm_harmonicity)**(params.norm_harmonicity))

            # Virtual Pitch
            self.virtualPitchContext.append(np.argmax(np.correlate(np.power(self.context[:,t],params.norm_harmonicity), SpecHarm,"full")))

        diffVirtualNotes = librosa.hz_to_note([self.fmin * (2**((i-len(SpecHarm)+dec+1)/BINS_PER_OCTAVE)) for i in self.virtualPitchContext] , cents = False)
        print(diffVirtualNotes[1:self.n_frames-1])


    def RoughnessContext(self):
        self.chrom_roughnessContext = np.zeros((self.n_bins,self.n_frames))
        for b1 in range(self.n_bins-1):
            for b2 in range(b1+1,self.n_bins):
                f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                freq = [f1, f2]
                freq.sort()
                if params.mod_rough == 'sethares + KK':
                    s = (1/2.27)*(np.log(params.β2/params.β1)/(params.β2-params.β1))/(freq[0]**(0.477))
                elif params.mod_rough == 'sethares':
                    s = 0.24/(0.021*freq[0] + 19)
                rug = np.exp(-params.β1*s*(freq[1]-freq[0]))-np.exp(-params.β2*s*(freq[1]-freq[0]))

                for t in range(self.n_frames):
                    self.chrom_roughnessContext[b1,t] += (self.context[b1,t] * self.context[b2,t]) * rug / 2
                    self.chrom_roughnessContext[b2,t] += (self.context[b1,t] * self.context[b2,t]) * rug / 2

        if params.norm_rugCtx:
            for t in range(self.n_frames-1):
                self.chrom_roughnessContext[:,t] = np.divide(self.chrom_roughnessContext[:,t], self.energyContext[t])
        self.roughnessContext = self.chrom_roughnessContext.sum(axis=0)


    def DiffConcordanceContext(self):
        self.chrom_diffConcordanceContext = np.zeros((self.n_bins,self.n_frames-1))
        if not params.theme_diffConcCtx:
            for t in range(self.n_frames-1):
                self.chrom_diffConcordanceContext[:,t] = np.multiply(self.context[:,t], self.chromSync[:,t+1])
                if params.norm_diffConcCtx:
                    self.chrom_diffConcordanceContext[:,t] /= np.sqrt(self.energyContext[t] * self.energy[t+1])
        else:
            for t in range(self.n_frames-1):
                self.chrom_diffConcordanceContext[:,t] = np.multiply(self.context[:,t], self.chromPistesSync[0][:,t+1])
                if params.norm_diffConcCtx:
                    self.chrom_diffConcordanceContext[:,t] /= np.sqrt(self.energyContext[t] * self.energyPistes[0][t+1])


        self.diffConcordanceContext = self.chrom_diffConcordanceContext.sum(axis=0)
        self.diffConcordanceContext[0]=0
        self.diffConcordanceContext[self.n_frames-2]=0


    def DiffRoughness(self):
        self.chrom_diffRoughness = np.zeros((self.n_bins,self.n_frames-1))
        if params.theme_diffRug:
            for b1 in range(self.n_bins):
                for b2 in range(self.n_bins):
                    f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                    f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                    freq = [f1, f2]
                    freq.sort()
                    if params.mod_rough == 'sethares + KK':
                        s = (1/2.27)*(np.log(params.β2/params.β1)/(params.β2-params.β1))/(freq[0]**(0.477))
                    elif params.mod_rough == 'sethares':
                        s = 0.24/(0.021*freq[0] + 19)
                    rug = np.exp(-params.β1*s*(freq[1]-freq[0]))-np.exp(-params.β2*s*(freq[1]-freq[0]))


                    for t in range(self.n_frames-1):
                        if params.rug_simpl:
                            self.chrom_diffRoughness[b1,t] += (self.chromSyncSimpl[b1,t] * self.chromPistesSyncSimpl[0][b2,t+1]) * rug / 2
                            self.chrom_diffRoughness[b2,t] += (self.chromSyncSimpl[b1,t] * self.chromPistesSyncSimpl[0][b2,t+1]) * rug / 2
                        else:
                            self.chrom_diffRoughness[b1,t] += (self.chromSync[b1,t] * self.chromPistesSync[0][b2,t+1]) * rug / 2
                            self.chrom_diffRoughness[b2,t] += (self.chromSync[b1,t] * self.chromPistesSync[0][b2,t+1]) * rug / 2

        else:
            for b1 in range(self.n_bins):
                for b2 in range(self.n_bins):
                    f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                    f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                    freq = [f1, f2]
                    freq.sort()
                    if params.mod_rough == 'sethares + KK':
                        s = (1/2.27)*(np.log(params.β2/params.β1)/(params.β2-params.β1))/(freq[0]**(0.477))
                    elif params.mod_rough == 'sethares':
                        s = 0.24/(0.021*freq[0] + 19)
                    rug = np.exp(-params.β1*s*(freq[1]-freq[0]))-np.exp(-params.β2*s*(freq[1]-freq[0]))


                    for t in range(self.n_frames-1):
                        if params.rug_simpl:
                            self.chrom_diffRoughness[b1,t] += (self.chromSyncSimpl[b1,t] * self.chromSyncSimpl[b2,t+1]) * rug / 2
                            self.chrom_diffRoughness[b2,t] += (self.chromSyncSimpl[b1,t] * self.chromSyncSimpl[b2,t+1]) * rug / 2
                        else:
                            self.chrom_diffRoughness[b1,t] += (self.chromSync[b1,t] * self.chromSync[b2,t+1]) * rug / 2
                            self.chrom_diffRoughness[b2,t] += (self.chromSync[b1,t] * self.chromSync[b2,t+1]) * rug / 2



        if params.norm_diffRug:
            if params.theme_diffRug:
                for t in range(self.n_frames-1):
                    self.chrom_diffRoughness[:,t] = np.divide(self.chrom_diffRoughness[:,t], np.sqrt(self.energy[t]*self.energyPistes[0][t+1]))
            else:
                for t in range(self.n_frames-1):
                    self.chrom_diffRoughness[:,t] = np.divide(self.chrom_diffRoughness[:,t], np.sqrt(self.energy[t]*self.energy[t+1]))

        self.diffRoughness = self.chrom_diffRoughness.sum(axis=0)
        self.diffRoughness[0]=0
        self.diffRoughness[self.n_frames-2]=0

    def DiffRoughnessContext(self):
        self.chrom_diffRoughnessContext = np.zeros((self.n_bins,self.n_frames-1))
        if params.theme_diffRugCtx:
            for b1 in range(self.n_bins):
                for b2 in range(self.n_bins):
                    f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                    f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                    freq = [f1, f2]
                    freq.sort()
                    if params.mod_rough == 'sethares + KK':
                        s = (1/2.27)*(np.log(params.β2/params.β1)/(params.β2-params.β1))/(freq[0]**(0.477))
                    elif params.mod_rough == 'sethares':
                        s = 0.24/(0.021*freq[0] + 19)
                    rug = np.exp(-params.β1*s*(freq[1]-freq[0]))-np.exp(-params.β2*s*(freq[1]-freq[0]))


                    for t in range(self.n_frames-1):
                        if params.rug_simpl:
                            self.chrom_diffRoughnessContext[b1,t] += (self.contextSimpl[b1,t] * self.chromPistesSyncSimpl[0][b2,t+1]) * rug / 2
                            self.chrom_diffRoughnessContext[b2,t] += (self.contextSimpl[b1,t] * self.chromPistesSyncSimpl[0][b2,t+1]) * rug / 2
                        else:
                            self.chrom_diffRoughnessContext[b1,t] += (self.context[b1,t] * self.chromPistesSync[0][b2,t+1]) * rug / 2
                            self.chrom_diffRoughnessContext[b2,t] += (self.context[b1,t] * self.chromPistesSync[0][b2,t+1]) * rug / 2

        else:
            for b1 in range(self.n_bins):
                for b2 in range(self.n_bins):
                    f1 = self.fmin*2**(b1/BINS_PER_OCTAVE)
                    f2 = self.fmin*2**(b2/BINS_PER_OCTAVE)
                    freq = [f1, f2]
                    freq.sort()
                    if params.mod_rough == 'sethares + KK':
                        s = (1/2.27)*(np.log(params.β2/params.β1)/(params.β2-params.β1))/(freq[0]**(0.477))
                    elif params.mod_rough == 'sethares':
                        s = 0.24/(0.021*freq[0] + 19)
                    rug = np.exp(-params.β1*s*(freq[1]-freq[0]))-np.exp(-params.β2*s*(freq[1]-freq[0]))


                    for t in range(self.n_frames-1):
                        if params.rug_simpl:
                            # self.chrom_diffRoughnessContext[b1,t] += (self.contextSimpl[b1,t] * self.chromSyncSimpl[b2,t+1]) * rug / 2
                            # self.chrom_diffRoughnessContext[b2,t] += (self.contextSimpl[b1,t] * self.chromSyncSimpl[b2,t+1]) * rug / 2
                            self.chrom_diffRoughnessContext[b1,t] += (self.contextSimpl[b1,t] * self.chromSyncSimpl[b2,t+1]) * rug / 2
                            self.chrom_diffRoughnessContext[b2,t] += (self.contextSimpl[b1,t] * self.chromSyncSimpl[b2,t+1]) * rug / 2
                        else:
                            self.chrom_diffRoughnessContext[b1,t] += (self.context[b1,t] * self.chromSync[b2,t+1]) * rug / 2
                            self.chrom_diffRoughnessContext[b2,t] += (self.context[b1,t] * self.chromSync[b2,t+1]) * rug / 2



        if params.norm_diffRugCtx:
            if params.theme_diffRugCtx:
                for t in range(self.n_frames-1):
                    self.chrom_diffRoughnessContext[:,t] = np.divide(self.chrom_diffRoughnessContext[:,t], np.sqrt(self.energyContext[t]*self.energyPistes[0][t+1]))
            else:
                for t in range(self.n_frames-1):
                    self.chrom_diffRoughnessContext[:,t] = np.divide(self.chrom_diffRoughnessContext[:,t], np.sqrt(self.energyContext[t]*self.energy[t+1]))

        self.diffRoughnessContext = self.chrom_diffRoughnessContext.sum(axis=0)
        self.diffRoughnessContext[0]=0
        self.diffRoughnessContext[self.n_frames-2]=0




    def ComputeDescripteurs(self, space = ['concordance','concordanceTot']):
        """Calcule les descripteurs indiqués dans 'space', puis les affiche"""

        dim = len(space)
        if 'concordance' in space: self.Concordance()
        if 'concordanceTot' in space: self.ConcordanceTot()
        if 'concordance3' in space: self.Concordance3()
        if 'tension' in space: self.Tension()
        if 'roughness' in space: self.Roughness()
        if 'tensionSignal' in space: self.TensionSignal()
        if 'roughnessSignal' in space: self.RoughnessSignal()
        if 'harmonicity' in space: self.Harmonicity()
        if 'crossConcordance' in space: self.CrossConcordance()
        if 'crossConcordanceTot' in space: self.CrossConcordanceTot()
        if 'harmonicChange' in space: self.HarmonicChange()
        if 'diffConcordance' in space: self.DiffConcordance()
        if 'diffRoughness' in space: self.DiffRoughness()
        if 'harmonicNovelty' in space: self.HarmonicNovelty()
        if 'harmonicityContext' in space: self.HarmonicityContext()
        if 'roughnessContext' in space: self.RoughnessContext()
        if 'diffConcordanceContext' in space: self.DiffConcordanceContext()
        if 'diffRoughnessContext' in space: self.DiffRoughnessContext()



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
                    # ax.set_xlim(min(xdescr[1:self.n_frames-1]), max(xdescr[1:self.n_frames-1])*6/5)
                    # ax.set_ylim(min(ydescr[1:self.n_frames-1]), max(ydescr[1:self.n_frames-1])*6/5)
                    ax.set_xlim(-0.03, 0.26)
                    ax.set_ylim(0.90, 0.99)
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

    def Affichage(self, space = ['concordance', 'concordanceTot'], begin = "first", end = "last", delete = []):
        #Plot de la recherche d'ONSETS
        # if title in params.dic_xcoords: self.onset_times_graph = np.array(params.dic_xcoords[title])
        if subTitle in params.dic_xcoords: self.onset_times_graph = np.array(params.dic_xcoords[subTitle])
        else: self.onset_times_graph = self.onset_times

        # Suppression des accords inutiles
        if len(delete) != 0:
            self.roughness = np.delete(self.roughness, delete, 0)
            self.chrom_roughness = np.delete(self.chrom_roughness, delete, 1)
            self.chromSyncDB = np.delete(self.chromSyncDB, delete, 1)
            self.n_frames = self.n_frames - len(delete)


        if params.plot_onsets:
            # fig, ax = plt.subplots(figsize=(13, 7))
            # # Partition
            # if params.plot_score & (len(self.score)!=0):
            #     img=mpimg.imread(self.score)
            #     score = plt.subplot(2,1,1)
            #     plt.axis('off')
            #     score.imshow(img)
            #     p = 1
            # else: p=0
            #
            # ax = plt.subplot(p+1,1,p+1)
            #
            # img = librosa.display.specshow(self.chromSyncDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph, cmap=cmap)
            # # img = librosa.display.specshow(librosa.amplitude_to_db(self.Chrom, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph, cmap=cmap)
            # # plt.title('Synchronised spectrum, β = {}, delay τ = {} s'.format(params.margin, T_att))
            # plt.title('Synchronised spectrum'.format(params.margin))
            # for t in self.onset_times_graph:
            #     ax.axvline(t, color = 'k',alpha=0.5, ls='--')
            # plt.axis('tight')
            # ax.get_xaxis().set_visible(False)
            # plt.tight_layout()


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
            librosa.display.specshow(self.chromSyncDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph, cmap=cmap)
            plt.tight_layout()

        #Plot de la décomposition en partie harmonique / partie percussive
        if params.plot_decompo_hpss & params.decompo_hpss:
            plt.figure(2,figsize=(13, 7))
            plt.subplot(3, 1, 1)
            librosa.display.specshow(librosa.amplitude_to_db(self.ChromNoHpss,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
            plt.title('Full cqt transform')

            plt.subplot(3, 1, 2)
            librosa.display.specshow(librosa.amplitude_to_db(self.Chrom,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
            plt.title('Harmonic part')

            plt.subplot(3, 1, 3)
            librosa.display.specshow(librosa.amplitude_to_db(self.ChromNoHpss - self.Chrom,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
            plt.title('Percussive part')
            plt.tight_layout()

        #Plot des pistes
        if params.plot_pistes:
            plt.figure(3,figsize=(13, 7.5))
            ax1 = plt.subplot(self.n_pistes,1,1)
            librosa.display.specshow(librosa.amplitude_to_db(self.chromPistesSyncSimpl[0], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph,cmap=cmap)
            for k in range(1, self.n_pistes):
                plt.subplot(self.n_pistes, 1, k+1, sharex=ax1)
                librosa.display.specshow(librosa.amplitude_to_db(self.chromPistesSyncSimpl[k], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph,cmap=cmap)
            plt.tight_layout()

        #Plot du contexte
        if params.plot_context:
            if len(self.context) == 0: self.context()
            if len(self.chrom_harmonicNovelty) == 0: self.HarmonicNovelty()

            plt.figure(8,figsize=(13, 7))
            plt.subplot(3, 1, 1)
            librosa.display.specshow(self.chromSyncDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph, cmap=cmap)
            plt.title('Synchronised cqt spectrum')

            plt.subplot(3, 1, 2)
            librosa.display.specshow(librosa.amplitude_to_db(self.context,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph,cmap=cmap)
            if isinstance(params.memory_size, str): title1 = 'Harmonic Context, cumulative memory'
            else : title1 = 'Harmonic Context, memory of {} chords'.format(int(params.memory_size))
            plt.title(title1)

            plt.subplot(3, 1, 3)
            librosa.display.specshow(librosa.amplitude_to_db(self.chrom_harmonicNovelty,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph,cmap=cmap)
            if isinstance(params.memory_size, str): title2 = 'Harmonic Novelties, cumulative memory'
            else : title2 = 'Harmonic Novelties, memory of {} chords'.format(int(params.memory_size))
            plt.title(title2)
            plt.tight_layout()

        #Plot des spectres simplifiés
        if params.plot_simple:
            # plt.figure(4,figsize=(13, 7.5))
            ########
            # ax1 = plt.subplot(2,1,1)
            # # librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times, cmap=cmap)
            # # plt.title('CQT spectrogram')
            #
            # # plt.subplot(2, 1, 1, sharex=ax1)
            # librosa.display.specshow(librosa.amplitude_to_db(self.chromSync, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph, cmap=cmap)
            # plt.title('Spectre syncronisé')
            #
            # plt.subplot(2, 1, 2, sharex=ax1)
            # librosa.display.specshow(librosa.amplitude_to_db(self.chromSyncSimpl, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap)
            # plt.title('Simplifié')
            # plt.tight_layout()

            ######
            # librosa.display.specshow(librosa.amplitude_to_db(self.chromSyncSimpl, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap)
            ######
            fig, ax = plt.subplots(figsize=(13, 7.5))
            img = librosa.display.specshow(librosa.amplitude_to_db(self.chromSyncSimpl, ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times_graph, cmap=cmap)
            # plt.title('Synchronised spectrum, β = {}, delay τ = {} s'.format(params.margin, T_att))
            plt.title('Partial detection on synchronised spectrum, β = {}, with delay, δ = {}'.format(params.margin, params.δ))
            for t in self.onset_times_graph:
                ax.axvline(t, color = 'k',alpha=0.5, ls='--')
            plt.axis('tight')
            plt.tight_layout()
            plt.show()

        #Plot les spectrogrammes
        if params.plot_chromDescr:

            #Construction de la liste des descripteurs avec Chrom
            spaceChrom = []
            for descr in space:
                if descr in ['concordance','concordance3','concordanceTot','roughness','crossConcordance','crossConcordanceTot','harmonicChange','diffConcordance']: spaceChrom.append(descr)


            dimChrom = len(space)
            times_plotChromDyn = [self.onset_times_graph[0]] + [t-0.25 for t in self.onset_times_graph[2:self.n_frames-1]] + [t+0.25 for t in self.onset_times_graph[2:self.n_frames-1]] + [self.onset_times_graph[self.n_frames]]
            times_plotChromDyn.sort()

            plt.figure(5,figsize=(13, 7.5))
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
                    if descr in ['roughness']:
                        librosa.display.specshow(getattr(self, 'chrom_'+descr), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap)
                    else:
                        librosa.display.specshow(librosa.amplitude_to_db(getattr(self, 'chrom_'+descr), ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap)

                    # Descripteurs dynamiques
                else:
                    Max  = np.amax(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2])
                    librosa.display.specshow(librosa.amplitude_to_db(np.insert(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2]/Max, range(self.n_frames-2),1, axis=1), ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=np.asarray(times_plotChromDyn), cmap=cmap)
                plt.title('Spectre de ' + descr)

            plt.tight_layout()

        #Plot les descripteurs harmoniques
        if params.plot_descr:
            dim = len(space)
            fig = plt.figure(6,figsize=(13, 7.5))

            # Partition
            if params.plot_score & (len(self.score)!=0):
                #plt.subplot(dim+1+s,1,s)
                img=mpimg.imread(self.score)
                score = plt.subplot(dim+1,1,1)
                plt.axis('off')
                score.imshow(img)
                plt.title(subTitle)

            else:
                ax1 = plt.subplot(dim+1,1,1)
                librosa.display.specshow(self.ChromDB, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.times,cmap=cmap)
                # plt.title(title +' '+ instrument)




            for k, descr in enumerate(space):
                if (k==0) & params.plot_score & (len(self.score)!=0):
                    ax1 = plt.subplot(dim+1,1,2)
                    ax1.get_xaxis().set_visible(False)
                else:
                    ax = plt.subplot(dim+1, 1, k+2, sharex=ax1)
                    ax.get_xaxis().set_visible(False)
                # Je remplace les valeurs nan par 0
                for i,val in enumerate(getattr(self,descr)):
                    if np.isnan(val): getattr(self,descr)[i] = 0

                if len(getattr(self, descr)) == self.n_frames:
                    plt.vlines(self.onset_times_graph[1:self.n_frames], min(getattr(self, descr)), max(getattr(self, descr)[1:(self.n_frames-1)]), color='k', alpha=0.9, linestyle='--')
                else:
                    plt.vlines(self.onset_times_graph[1:self.n_frames-1], min(getattr(self, descr)), max(getattr(self, descr)[1:(self.n_frames-1)]), color='k', alpha=0.9, linestyle='--')
                plt.xlim(self.onset_times_graph[0],self.onset_times_graph[-1])
                if not all(x>=0 for x in getattr(self, descr)[1:(self.n_frames-1)]):
                    plt.hlines(0,self.onset_times_graph[0], self.onset_times_graph[self.n_frames], alpha=0.5, linestyle = ':')

                # Legend
                context = ''
                norm = ''
                if params.plot_norm and (descr in params.dic_norm ): norm = '\n' + params.dic_norm[descr]
                if descr in ['harmonicNovelty', 'harmonicityContext','roughnessContext','diffConcordanceContext','diffRoughnessContext'] :
                    if params.memory_size>=2: context = '\n' + 'Memory: {} chords, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
                    else: context = '\n' + 'Memory: {} chord, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)

                # Descripteurs statiques
                if len(getattr(self, descr)) == self.n_frames:
                    plt.hlines(getattr(self, descr)[1:(self.n_frames-1)], self.onset_times_graph[1:(self.n_frames-1)], self.onset_times_graph[2:self.n_frames],color=['b','r','g','c','m','y','b','r','g'][k] , label=descr[0].upper() + descr[1:] + norm + context)
                # Descripteurs dynamiques
                elif len(getattr(self, descr)) == (self.n_frames-1):
                    if descr == 'diffRoughnessContext': plt.plot(self.onset_times_graph[2:(self.n_frames-1)], getattr(self, descr)[1:(self.n_frames-2)],['b','r','g','c','m','y','b','r','g'][k]+'o', label='DiffRoughness' + norm)
                    else: plt.plot(self.onset_times_graph[2:(self.n_frames-1)], getattr(self, descr)[1:(self.n_frames-2)],['b','r','g','c','m','y','b','r','g'][k]+'o', label=(descr[0].upper() + descr[1:]) + norm)
                    plt.hlines(getattr(self, descr)[1:(self.n_frames-2)], [t-0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], [t+0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], color=['b','r','g','c','m','y','b','r','g'][k], alpha=0.9, linestyle=':'  )
                    # plt.plot(self.onset_times_graph[2:(self.n_frames-1)], [0.35, 0.21, 0.34, 0.23],['b','r','g','c','m','y','b','r','g'][1]+'o', label = 'Octave up')#label=(descr[0].upper() + descr[1:]) + norm + context)
                    # plt.hlines([0.35, 0.21, 0.34, 0.23], [t-0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], [t+0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], color=['b','r','g','c','m','y','b','r','g'][1], alpha=0.9, linestyle=':'  )
                    # plt.hlines([0.61,0.47,0.59, 0.49], [t-0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], [t+0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], color=['b','r','g','c','m','y','b','r','g'][2], alpha=0.9, linestyle=':'  )
                    # plt.plot(self.onset_times_graph[2:(self.n_frames-1)], [0.61,0.47,0.59, 0.49],['b','r','g','c','m','y','b','r','g'][2]+'o',label = 'Fourth down')#label=(descr[0].upper() + descr[1:]) + norm + context)

                # plt.title('DiffRoughness, normalised')
                plt.legend(frameon=True, framealpha=0.75)

            plt.tight_layout()

        #Plot descriptogramme + valeur numérique
        if params.plot_OneDescr:

            descr = space[0]
            plt.figure(7,figsize=(10, 7.5))

            ############################

            # Partition
            if params.plot_score & (len(self.score)!=0):
                img=mpimg.imread(self.score)
                score = plt.subplot(3,1,1)
                plt.axis('off')
                score.imshow(img)
                p = 1
            else: p=0

            ax1 = plt.subplot(p+2,1,p+1)

            # Descripteurs statiques
            if len(getattr(self, descr)) == self.n_frames:
                if descr in ['concordanceTot', 'concordance3','roughness']:
                    librosa.display.specshow(getattr(self, 'chrom_'+descr), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap)
                elif descr == 'harmonicity':
                    librosa.display.specshow(np.power(getattr(self, 'chrom_'+descr),4)[0:4*BINS_PER_OCTAVE], bins_per_octave=BINS_PER_OCTAVE, fmin=f_corr_min, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap)
                else:
                    librosa.display.specshow(librosa.amplitude_to_db(getattr(self, 'chrom_'+descr)[0:int(5*self.n_bins/6),:], ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times_graph, cmap=cmap,sr = self.sr)

                # Descripteurs dynamiques
            else:
                times_plotChromDyn = [self.onset_times_graph[0]] + [t-0.75 for t in self.onset_times_graph[2:self.n_frames-1]] + [t+0.75 for t in self.onset_times_graph[2:self.n_frames-1]] + [self.onset_times_graph[self.n_frames]]
                times_plotChromDyn.sort()
                Max  = np.amax(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2])
                if descr == 'diffRoughnessContext':
                    librosa.display.specshow(np.insert(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2]/Max, range(self.n_frames-2),1, axis=1), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=np.asarray(times_plotChromDyn),cmap=cmap)
                else:
                    librosa.display.specshow(librosa.amplitude_to_db(np.insert(getattr(self, 'chrom_'+descr)[:,1:self.n_frames-2]/Max, range(self.n_frames-2),1, axis=1), ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=np.asarray(times_plotChromDyn),cmap=cmap)

            for t in self.onset_times_graph:
                ax1.axvline(t, color = 'k',alpha=0.5, ls='--')
            if descr == 'harmonicity':
                plt.title('Virtual pitch spectrum')
            else:
                # plt.title('DiffRoughness Spectrum')
                plt.title(descr[0].upper()+descr[1:]+' spectrum')
            plt.xlim(self.onset_times_graph[0],self.onset_times_graph[-1])
            ax1.get_xaxis().set_visible(False)


            # Plot Descr
            ax2 = plt.subplot(p+2, 1, p+2)
            if len(getattr(self, descr)) == self.n_frames:
                plt.vlines(self.onset_times_graph[1:self.n_frames], min(getattr(self, descr)), max(getattr(self, descr)), color='k', alpha=0.9, linestyle='--')
            else:
                plt.vlines(self.onset_times_graph[1:self.n_frames-1], min(getattr(self, descr)),max(getattr(self, descr)), color='k', alpha=0.9, linestyle='--')
            if not all(x>=0 for x in getattr(self, descr)):
                plt.hlines(0,self.onset_times_graph[0], self.onset_times_graph[self.n_frames], alpha=0.5, linestyle = ':')

            # Legend
            context = ''
            norm = ''
            par = ''
            if params.plot_norm and (descr in params.dic_norm ): norm = '\n' + params.dic_norm[descr]
            if descr in ['harmonicNovelty', 'harmonicityContext','roughnessContext','diffConcordanceContext','diffRoughnessContext'] :
                if params.memory_size>=2: context = '\n' + 'Memory: {} chords, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
                else: context = '\n' + 'Memory: {} chord'.format(params.memory_size)
            if descr in ['harmonicity']:
                par = '\n{} partials'.format(params.κ)

                # Descripteurs statiques
            if len(getattr(self, descr)) == self.n_frames:
                plt.hlines(getattr(self, descr)[1:(self.n_frames-1)], self.onset_times_graph[1:(self.n_frames-1)], self.onset_times_graph[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][1], label= descr[0].upper() + descr[1:] + norm + context + par)
                # Descripteurs dynamiques
            elif len(getattr(self, descr)) == (self.n_frames-1):
                plt.plot(self.onset_times_graph[2:(self.n_frames-1)], getattr(self, descr)[1:(self.n_frames-2)],['b','r','g','c','m','y','b','r','g'][0]+'o')
                plt.hlines(getattr(self, descr)[1:(self.n_frames-2)], [t-0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], [t+0.5 for t in self.onset_times_graph[2:(self.n_frames-1)]], color=['b','r','g','c','m','y','b','r','g'][0], alpha=0.9, linestyle=':',label = descr[0].upper() + descr[1:] + norm + context)
            plt.xlim(self.onset_times_graph[0],self.onset_times_graph[-1])
            # plt.ylim(bottom=0)
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            ax2.get_xaxis().set_visible(False)
            plt.legend(frameon=True, framealpha=0.75)
            plt.tight_layout()



        #Plot représentations abstraites
        if params.plot_abstr:
            if len(space)==2 :
                color = params.color_abstr
                l1 = getattr(self, space[0])[1:len(getattr(self, space[0]))-1]
                l2 = getattr(self, space[1])[1:len(getattr(self, space[1]))-1]

                #Si un descripteur statique et un descripteur dynamique
                if len(l1)<len(l2) : l2.pop(0)
                elif len(l1)>len(l2) : l1.pop(0)

                #Tronquage
                if isinstance(end,int):
                    l1= l1[0:end]
                    l2= l2[0:end]

                plt.figure(8)
                ax = plt.subplot()
                if params.link_abstr: plt.plot(l1, l2, color+'--')
                plt.plot(l1, l2, color+'o')
                for i in range(len(l1)):
                    ax.annotate(' {}'.format(i+1), (l1[i], l2[i]), color=params.color_abstr_numbers)
                plt.xlabel(space[0][0].upper() + space[0][1:])
                plt.ylabel(space[1][0].upper() + space[1][1:])
                plt.title(title +' '+instrument + ' (' + space[0][0].upper() + space[0][1:] + ', ' + space[1][0].upper() + space[1][1:] + ')')

            else:
                color = params.color_abstr
                l1 = getattr(self, space[0])[1:len(getattr(self, space[0]))-1]
                l2 = getattr(self, space[1])[1:len(getattr(self, space[0]))-1]
                l3 = getattr(self, space[2])[1:len(getattr(self, space[0]))-1]
                fig = plt.figure(9)
                ax = fig.add_subplot(111, projection='3d')
                if params.link_abstr: plt.plot(l1, l2, l3, color+'--')
                for i in range(len(l1)):
                    ax.scatter(l1[i], l2[i], l3[i], c=color, marker='o')
                    ax.text(l1[i], l2[i], l3[i], i+1, color=params.color_abstr_numbers)
                ax.set_xlabel(space[0][0].upper() + space[0][1:])
                ax.set_ylabel(space[1][0].upper() + space[1][1:])
                ax.set_zlabel(space[2][0].upper() + space[2][1:])
                ax.set_title(title +' '+instrument + ' (' + space[0][0].upper() + space[0][1:] + ', ' + space[1][0].upper() + space[1][1:] + ', ' + space[2][0].upper() + space[2][1:] + ')')


        if params.plot_compParam:
            # Représentation des nouveautés, comparaison des échelles de mémoire
            with open ('nouv0', 'rb') as fp:
                nouv0 = pickle.load(fp)
            with open ('nouv1', 'rb') as fp:
                nouv1 = pickle.load(fp)
            with open ('nouv2', 'rb') as fp:
                nouv2 = pickle.load(fp)
            with open ('nouv3', 'rb') as fp:
                nouv3 = pickle.load(fp)
            with open ('nouv4', 'rb') as fp:
                nouv4 = pickle.load(fp)
            with open ('nouvFull', 'rb') as fp:
                nouvFull = pickle.load(fp)

            plt.figure(9,figsize=(13, 7))
            img=mpimg.imread(self.score)
            score = plt.subplot(2,1,1)
            plt.axis('off')
            score.imshow(img)
            plt.title(title +' '+instrument)

            plt.subplot(2, 1, 2)
            plt.vlines(self.onset_times_graph[1:self.n_frames], 0, 1, color='k', alpha=0.9, linestyle='--')
            plt.hlines(nouv0[1:(self.n_frames-1)], self.onset_times_graph[1:(self.n_frames-1)], self.onset_times_graph[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][2], label='Memory: 0 chord')
            # plt.hlines(nouv1[1:(self.n_frames-1)], self.onset_times_graph[1:(self.n_frames-1)], self.onset_times_graph[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][3], label='Memory: 1 chord')
            # plt.hlines(nouv2[1:(self.n_frames-1)], self.onset_times_graph[1:(self.n_frames-1)], self.onset_times_graph[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][4], label='Memory: 2 chords')
            plt.hlines(nouv3[1:(self.n_frames-1)], self.onset_times_graph[1:(self.n_frames-1)], self.onset_times_graph[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][5], label='Memory: 3 chords')
            # plt.hlines(nouv4[1:(self.n_frames-1)], self.onset_times_graph[1:(self.n_frames-1)], self.onset_times_graph[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][6], label='Memory: 4 chords')
            plt.hlines(nouvFull[1:(self.n_frames-1)], self.onset_times_graph[1:(self.n_frames-1)], self.onset_times_graph[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][1], label='Memory: All chords')

            plt.legend(frameon=True, framealpha=0.75)
            plt.tight_layout()


        plt.show()





    def Points(self, space = ['concordance', 'concordanceTot']):

        L = []
        for descr in space:
            if isinstance(getattr(self,descr), list): L.append(getattr(self,descr)[1:-1])
            else : L.append(getattr(self,descr).tolist()[1:-1])

        # Si à la fois descripteurs statiques et dynamiques dans space, alors on réduit la longueur des listes de descripteurs statiques en considérant l'évolution du descripteur statique
        T = min(map(len,L))
        for i in range(len(space)):
            if len(L[i])>T:
                for t in range(len(L[i])-1):
                    L[i][t] = L[i][t+1] - L[i][t]
                L[i].pop(-1)

        Points = np.asarray(L)
        return Points




    def Sort(self, space = ['concordance']):
        descr = space[0]
        L = getattr(self,descr)[1:self.n_frames-1]
        indices, L_sorted = zip(*sorted(enumerate(L), key=itemgetter(1), reverse=params.sorted_reverse))

        if params.plot_sorted:
            if params.sorted_reverse: croiss = 'décroissante'
            else: croiss = 'croissante'
            sorted_score = 'Exemples/'+ title +'-'+descr+'-score.png'
            plt.figure(1,figsize=(13, 7))

            img=mpimg.imread(sorted_score)
            score = plt.subplot(2,1,1)
            plt.axis('off')
            score.imshow(img)
            plt.title(descr[0].upper() + descr[1:] +' '+croiss)

            # Plot Descr
            plt.subplot(2, 1, 2)
            plt.vlines(self.onset_times_graph[1:self.n_frames], min(getattr(self, descr)), max(getattr(self, descr)), color='k', alpha=0.9, linestyle='--')

                # Descripteurs statiques
            if len(getattr(self, descr)) == self.n_frames:
                plt.hlines(L_sorted, self.onset_times_graph[1:(self.n_frames-1)], self.onset_times_graph[2:self.n_frames], color=['b','r','g','c','m','y','b','r','g'][1], label=descr[0].upper() + descr[1:])

            plt.legend(frameon=True, framealpha=0.75)
            plt.show()
        else : print(indices)



# PARAMETRES
type_Temporal = params.type_Temporal
type_Normalisation = params.type_Normalisation




def Construction_Points(liste_timbres_or_scores, title, space, Notemin, Notemax, dic, filename = 'Points.npy', score = [], instrument = 'Organ', name_OpenFrames = '', duration = None, sr = 44100, share_onsets = True, liste_no_verticality = []):
    # Chargement de onset_frames
    def OpenFrames(title):
        #1 - Avec Sonic Visualiser
        if os.path.exists('Onset_given_'+title+'.txt'):
            onsets = []
            with open('Onset_given_'+title+'.txt','r') as f:
                for line in f:
                    l = line.split()
                    onsets.append(float(l[0]))
            onset_times = np.asarray(onsets)
            onset_frames = librosa.time_to_frames(onset_times, sr=sr, hop_length = STEP)

        #2 - À partir de la partition en musicxml
        elif os.path.exists('Onset_given_'+title+'_score'):
            with open('Onset_given_'+title+'_score', 'rb') as f:
                onsets = pickle.load(f)
                onset_times = np.asarray(onsets)
                onset_frames = librosa.time_to_frames(onset_times, sr=sr, hop_length = STEP)

        #3 - Avec ma méthode de calcul automatique
        elif os.path.exists('Onset_given_'+title):
            with open('Onset_given_'+title, 'rb') as f:
                onset_frames = pickle.load(f)

        #4 - Pas d'onsets préchargés
        else: onset_frames = []
        # for i,time in enumerate(onset_frames)
        return onset_frames
    if share_onsets:
        if len(name_OpenFrames)==0: onset_frames = OpenFrames(title)
        else: onset_frames = OpenFrames(name_OpenFrames)

    global distribution
    if title in params.dic_distribution: distribution = params.dic_distribution[title]
    else : distribution = params.distribution
    Points = []

    if params.one_track:
        for instrument in liste_timbres_or_scores:
            # CHARGEMENT DES SONS ET DE LA PARTITION
            y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'.wav', duration = duration)
            if distribution == 'voix':
                y1, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-Basse.wav', duration = duration)
                y2, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-Alto.wav', duration = duration)
                y3, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-Soprano.wav', duration = duration)
                y4, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-Tenor.wav', duration = duration)
            elif distribution == 'themeAcc':
                y1, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-theme.wav', duration = duration)
                y2, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-acc.wav', duration = duration)

            # CRÉATION DE L'INSTANCE DE CLASSE
            if distribution == 'record':
                S = SignalSepare(y, sr, [], Notemin, Notemax,onset_frames, score = score,instrument = instrument)
            elif distribution == 'voix':
                S = SignalSepare(y, sr, [y1,y2,y3,y4], Notemin, Notemax,onset_frames, score = score,instrument = instrument)
            elif distribution == 'themeAcc':
                S = SignalSepare(y, sr, [y1,y2], Notemin, Notemax,onset_frames, score = score,instrument = instrument)

            S.DetectionOnsets()
            S.Clustering()
            S.Context()
            if params.simpl: S.SimplifySpectrum()
            S.ComputeDescripteurs(space = space)

            # CRÉATION DE POINTS
            Points.append(S.Points(space))
            print(instrument + ': OK')

    if params.compare_instruments:
        for instrument in liste_timbres_or_scores:
            # CHARGEMENT DES SONS ET DE LA PARTITION
            y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'_T{}'.format(dic[instrument])+'.wav', duration = duration)
            y1, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'_T{}'.format(dic[instrument])+'-Basse.wav', duration = duration)
            y2, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'_T{}'.format(dic[instrument])+'-Alto.wav', duration = duration)
            y3, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'_T{}'.format(dic[instrument])+'-Soprano.wav', duration = duration)
            y4, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'_T{}'.format(dic[instrument])+'-Tenor.wav', duration = duration)

            # CRÉATION DE L'INSTANCE DE CLASSE
            if distribution == 'record':
                S = SignalSepare(y, sr, [], Notemin, Notemax,onset_frames, score = score,instrument = instrument)
            elif distribution == 'voix':
                S = SignalSepare(y, sr, [y1,y2,y3,y4], Notemin, Notemax,onset_frames, score = score,instrument = instrument)
            elif distribution == 'themeAcc':
                S = SignalSepare(y, sr, [y1,y2], Notemin, Notemax,onset_frames, score = score,instrument = instrument)
            S.DetectionOnsets()
            S.Clustering()
            S.Context()
            if params.simpl: S.SimplifySpectrum()
            S.ComputeDescripteurs(space = space)

            # CRÉATION DE POINTS
            Points.append(S.Points(space))
            print(instrument + ': OK')

    if params.compare_scores:
        for score in liste_timbres_or_scores:
            # CHARGEMENT DES SONS ET DE LA PARTITION
            y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+ title +'{}.wav'.format(dic[score]), duration = duration)
            y1, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'{}-Basse.wav'.format(dic[score]), duration = duration)
            y2, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'{}-Alto.wav'.format(dic[score]), duration = duration)
            y3, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'{}-Soprano.wav'.format(dic[score]), duration = duration)
            y4, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'{}-Tenor.wav'.format(dic[score]), duration = duration)

            # CRÉATION DE L'INSTANCE DE CLASSE
            if distribution == 'record':
                S = SignalSepare(y, sr, [], Notemin, Notemax,onset_frames, delOnsets, addOnsets, score = score,instrument = instrument)
            elif distribution == 'voix':
                S = SignalSepare(y, sr, [y1,y2,y3,y4], Notemin, Notemax,onset_frames, delOnsets, addOnsets, score = score,instrument = instrument)
            elif distribution == 'themeAcc':
                S = SignalSepare(y, sr, [y1,y2], Notemin, Notemax,onset_frames, delOnsets, addOnsets, score = score,instrument = instrument)
            S.DetectionOnsets()
            S.Clustering()
            S.Context()
            if params.simpl: S.SimplifySpectrum()
            S.ComputeDescripteurs(space = space)

            # CRÉATION DE POINTS
            Points.append(S.Points(space))
            print(score + ': OK')

    if params.compare:
        for subtitle in liste_timbres_or_scores:
            # CHARGEMENT DES SONS ET DE LA PARTITION
            y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/' + subtitle +'.wav', duration = params.dic_duration[title], sr = None)
            y1, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/' + subtitle  +'-Soprano.wav', duration = params.dic_duration[title], sr = None)
            y2, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/' + subtitle  +'-Alto.wav', duration = params.dic_duration[title], sr = None)
            y3, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/' + subtitle  +'-Tenor.wav', duration = params.dic_duration[title], sr = None)
            y4, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/' + subtitle  +'-Bass.wav', duration = params.dic_duration[title], sr = None)
            # y5, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/' + subtitle  +'-Baryton.wav', duration = duration, sr = None)

            if not share_onsets:
                onset_frames = OpenFrames(subtitle)

            # CRÉATION DE L'INSTANCE DE CLASSE
            if distribution == 'record':
                S = SignalSepare(y, sr, [], Notemin, Notemax,onset_frames, score = score,instrument = instrument)
            elif distribution == 'voix':
                S = SignalSepare(y, sr, [y1,y2,y3,y4], Notemin, Notemax,onset_frames, score = score,instrument = instrument)
            elif distribution == 'themeAcc':
                S = SignalSepare(y, sr, [y1,y2], Notemin, Notemax,onset_frames, score = score,instrument = instrument)
            S.DetectionOnsets()
            S.Clustering()
            S.Context()
            if params.simpl: S.SimplifySpectrum()
            S.ComputeDescripteurs(space = space)

            # CRÉATION DE POINTS
            Points.append(S.Points(space))
            print(subtitle + ': OK')

    if len(liste_no_verticality)!=0:
        for i in liste_no_verticality:
            Points[0] = np.insert(Points[0],i, None,1)
    print(Points[0].shape, Points[1].shape)
    Points = np.asarray(Points)
    print('Taille : {}'.format(Points.shape))
    np.save(filename, Points) # save
    np.save('Onset_times_'+filename,S.onset_times)


# Fonction qui normalise la matrice Points
def Normalise(Points, liste_timbres_or_scores, dic, type_Normalisation = type_Normalisation):
    ind_instrument = [dic[instrument]-1 for instrument in liste_timbres_or_scores]
    Points = Points[ind_instrument]
    if type_Normalisation == 'by timbre':
        max = np.nanmax(Points, axis = (0,2))
        for descr in range(Points.shape[1]):
            Points[:,descr,:] /= max[descr]
    elif type_Normalisation == 'by curve':
        max = np.nanmax(Points, axis = 2)
        for timbre in range(Points.shape[0]):
            for descr in range(Points.shape[1]):
                Points[timbre,descr,:] /= max[timbre,descr]
    return Points

# Fonction qui calcule la matrice des écart-types sur tous les timbres
def Dispersion(Points,type_Temporal = type_Temporal):
    if type_Temporal == 'static':
        Disp = np.std(Points,axis = 0)
    elif type_Temporal == 'differential':
        Points_diff = np.zeros((Points.shape[0],Points.shape[1],Points.shape[2]-1))
        for i in range(Points.shape[2]-1):
            Points_diff[:,:,i] = Points[:,:,i+1]-Points[:,:,i]
        Disp = np.std(Points_diff,axis = 0)
    Disp_by_descr = np.mean(Disp, axis = 1)
    Disp_by_time = np.linalg.norm(Disp, axis = 0)
    return Disp, Disp_by_descr,Disp_by_time


def Inerties(Points, type_Temporal = type_Temporal):
    if type_Temporal == 'static':
        Inertie_tot = np.std(Points, axis = (0,2))
        Mean = np.mean(Points,axis = 0)
    elif type_Temporal == 'differential':
        Points_diff = np.zeros((Points.shape[0],Points.shape[1],Points.shape[2]-1))
        for i in range(Points.shape[2]-1):
            Points_diff[:,:,i] = Points[:,:,i+1]-Points[:,:,i]
        Inertie_tot = np.std(Points_diff, axis = (0,2))
        Mean = np.mean(Points_diff, axis = 0)
    Inertie_inter = np.std(Mean, axis = 1)
    return Inertie_tot, Inertie_inter

# Fonction qui trie les descripteurs en fonction du minimum de dispersion
def MinimizeDispersion(Disp_by_descr, space):
    disp_sorted = np.sort(Disp_by_descr)
    descr_sorted = [space[i] for i in np.argsort(Disp_by_descr)]
    return descr_sorted, disp_sorted

# Fonction qui trie les descripteurs en fonction du minimum de dispersion
def MaximizeSeparation(Inertie_tot, Inertie_inter, space):
    d = len(space)
    sep_matrix = np.zeros((d,d))
    for i in range(1,d):
        for j in range(i):
            sep_matrix[i,j] = np.inner(Inertie_inter[[i,j]], Inertie_inter[[i,j]]) / np.inner(Inertie_tot[[i,j]], Inertie_tot[[i,j]])
    ind = np.unravel_index(np.argmax(sep_matrix, axis=None), sep_matrix.shape)
    return [space[ind[0]], space[ind[1]]]


def Clustered(Points, spacePlot, space, type_Temporal = type_Temporal):
    ind_descr = [space.index(descr) for descr in spacePlot]
    if type_Temporal == 'static':
        Points_sub = Points[:,ind_descr]
    if type_Temporal == 'differential':
        Points_diff = np.zeros((Points.shape[0],Points.shape[1],Points.shape[2]-1))
        for i in range(Points.shape[2]-1):
            Points_diff[:,:,i] = Points[:,:,i+1]-Points[:,:,i]
        Points_sub = Points_diff[:,ind_descr]
    disp_traj = np.sum(np.linalg.norm(np.std(Points_sub,axis = 0), axis = 0))
    inertie_inter = np.std(np.mean(Points_sub,axis = 0), axis = 1)
    inertie_tot = np.std(Points_sub, axis = (0,2))
    sep = np.inner(inertie_inter, inertie_inter) / np.inner(inertie_tot, inertie_tot)
    print('Dispersion : {} \nSeparation : {}'.format(disp_traj, sep))


# Visualisation
def Visualize(Points, descr, space, liste_timbres_or_scores, dic, type_Temporal = type_Temporal, type = 'abstract', simultaneity = False, score = None, onset_times = None, onset_times_graph = None, liste_Points = [], erase = None, liste_annot = []):
    # Liste des descripteurs de context
    liste_descrContext = ['energyContext','harmonicChange','harmonicNovelty', 'harmonicityContext','roughnessContext','diffConcordanceContext','diffRoughnessContext']
    def Erase(l,erase):
        erase.sort(reverse = True)
        for i in erase:
            del l[i]
        return l


    if type == 'abstract':
        dim1 = space.index(descr[0])
        dim2 = space.index(descr[1])

        # Fonction qui renvoie True si deux listes ont une intersection commune
        def intersect(lst1, lst2):
            inter = False
            i = 0
            while (not inter) & (i<len(lst1)):
                if (lst1[i] in lst2): inter = True
                i += 1
            return inter

        # Détermination de la présence simultanée de descripteurs statiques et dynamiques dans space, et le cas échéant attribution du suffixe 'evolution' aux descr stat
        suff0, suff1 = '',''
        if intersect(space,spaceStat) & intersect(space,spaceDyn):
            if descr[0] in spaceStat: suff0 = ' evolution'
            if descr[1] in spaceStat: suff1 = ' evolution'

        plt.figure(figsize=(8, 7))
        ax = plt.subplot()


        if type_Temporal =='static':
            if len(liste_Points) > 0:
                for k,points in enumerate(liste_Points):
                    for timbre, instrument in enumerate(liste_timbres_or_scores):
                        if dic[instrument] <= 10: ls = '--'
                        else: ls = ':'
                        if params.visualize_trajectories:
                            if params.one_track and params.compare_contexts:
                                if instrument[1]>=2: label = '\n' + 'Memory: {} chords, decr = {}'.format(instrument[1], instrument[2])
                                else: label = '\n' + 'Memory: {} chord, decr = {}'.format(instrument[1], instrument[2])
                            else: label = instrument
                            plt.plot(points[timbre,dim1,:].tolist(), points[timbre,dim2,:].tolist(), color ='C{}'.format(k),ls = ls, marker = 'o', label = instrument)
                        if params.visualize_time_grouping:
                            for t in range(len(points[timbre,dim1,:])):
                                plt.plot(points[timbre,dim1,:].tolist()[t], points[timbre,dim2,:].tolist()[t], color ='C{}'.format(t),ls = ls, marker = 'o')
                        for t in range(len(points[timbre,dim1,:].tolist())):
                            ax.annotate(' {}'.format(t+1), (points[timbre,dim1,:][t], points[timbre,dim2,:][t]), color='black')

            else:
                for timbre, instrument in enumerate(liste_timbres_or_scores):
                    if dic[instrument] <= 10: ls = '--'
                    else: ls = ':'
                    if params.visualize_trajectories:
                        if params.one_track and params.compare_contexts:
                            if instrument[1]>=2: label = '\n' + 'Memory: {} chords, decr = {}'.format(instrument[1], instrument[2])
                            else: label = '\n' + 'Memory: {} chord, decr = {}'.format(instrument[1], instrument[2])
                        else: label = instrument
                        plt.plot(Points[timbre,dim1,:].tolist(), Points[timbre,dim2,:].tolist(), color ='C{}'.format(dic[instrument]-1),ls = ls, marker = 'o', label = instrument)
                    if params.visualize_time_grouping:
                        for t in range(len(Points[timbre,dim1,:])):
                            plt.plot(Points[timbre,dim1,:].tolist()[t], Points[timbre,dim2,:].tolist()[t], color ='C{}'.format(t),ls = ls, marker = 'o')
                    for t in range(len(Points[timbre,dim1,:].tolist())):
                        if len(liste_annot)==0:
                            ax.annotate(' {}'.format(t+1), (Points[timbre,dim1,:][t], Points[timbre,dim2,:][t]), color='black')
                        else:
                            ax.annotate('  {}'.format(liste_annot[t]), (Points[timbre,dim1,:][t], Points[timbre,dim2,:][t]), color='black')

            if not all(x>=0 for x in Points[:,dim1,:].flatten()):
                plt.vlines(0,np.amin(Points[:,dim2,:]), np.amax(Points[:,dim2,:]), alpha=0.5, linestyle = ':')
            if not all(x>=0 for x in Points[:,dim2,:].flatten()):
                plt.hlines(0,np.amin(Points[:,dim1,:]), np.amax(Points[:,dim1,:]), alpha=0.5, linestyle = ':')

            #Legend
            context0, context1 = '',''
            norm0, norm1 = '',''
            if params.plot_norm and (descr[0] in params.dic_norm ): norm0 = ', ' + params.dic_norm[descr[0]]
            if params.plot_norm and (descr[1] in params.dic_norm ): norm1 = ', ' + params.dic_norm[descr[1]]
            if not params.compare_contexts:
                if descr[0] in liste_descrContext:
                    if params.memory_size>=2: context0 = ', '+'Memory: {} chords, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
                    else: context0 = ', '+'Memory: {} chord, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
                if descr[1] in liste_descrContext:
                    if params.memory_size>=2: context1 = ', '+'Memory: {} chords, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
                    else: context1 = ', '+'Memory: {} chord, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
            plt.xlabel(descr[0][0].upper() + descr[0][1:] + suff0 + norm0)# + context0)
            plt.ylabel(descr[1][0].upper() + descr[1][1:] + suff1 + norm1)# + context1)
            # plt.xlabel('Concordance différentielle, Normalisée')
            # plt.ylabel('Rugosité différentielle, Non normalisée')

            if params.one_track and not params.compare_contexts:
                plt.title(title + ' (' + descr[0][0].upper() + descr[0][1:] + suff0 + ', ' + descr[1][0].upper() + descr[1][1:] + suff1 + ')\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')
            else:
                if params.compare_instruments: goal = 'Timbre comparaison'
                elif params.compare_scores: goal = 'Score comparaison'
                elif params.one_track and params.compare_contexts: goal = 'Context comparaison'
                elif params.compare: goal = 'Comparaison des accords'
                if type_Normalisation == 'by curve':
                    plt.title(goal + '\n' + title + ' (' + descr[0][0].upper() + descr[0][1:] + suff0 + ', ' + descr[1][0].upper() + descr[1][1:] + suff1 + ')\n' + 'Normalisation curve by curve' + '\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')
                else:
                    plt.title(goal + '\n' + title + ' (' + descr[0][0].upper() + descr[0][1:] + suff0 + ', ' + descr[1][0].upper() + descr[1][1:] + suff1 + ')\n' + 'Normalisation on all the curves ' + '\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')





        elif type_Temporal =='differential':
            # Construction de la matrice Points_diff
            Points_diff = np.zeros((Points.shape[0],Points.shape[1],Points.shape[2]-1))
            for i in range(Points.shape[2]-1):
                Points_diff[:,:,i] = Points[:,:,i+1]-Points[:,:,i]
            for timbre, instrument in enumerate(liste_timbres_or_scores):
                if dic[instrument] <= 10: ls = '--'
                else: ls = ':'
                if params.visualize_trajectories:
                    if params.one_track and params.compare_contexts:
                        if instrument[1]>=2: label = '\n' + 'Memory: {} chords, decr = {}'.format(instrument[1], instrument[2])
                        else: label = '\n' + 'Memory: {} chord, decr = {}'.format(instrument[1], instrument[2])
                    else: label = instrument
                    plt.plot(Points_diff[timbre,dim1,:].tolist(), Points_diff[timbre,dim2,:].tolist(), color ='C{}'.format(dic[instrument]-1),ls = ls, marker = 'o', label = instrument)
                if params.visualize_time_grouping:
                    for t in range(len(Points_diff[timbre,dim1,:])):
                        plt.plot(Points_diff[timbre,dim1,:].tolist()[t], Points_diff[timbre,dim2,:].tolist()[t], color ='C{}'.format(t),ls = ls, marker = 'o')
                for t in range(len(Points_diff[timbre,dim1,:].tolist())):
                    ax.annotate(' {}'.format(t+1), (Points_diff[timbre,dim1,:][t], Points_diff[timbre,dim2,:][t]), color='black')
            if not all(x>=0 for x in Points_diff[:,dim1,:].flatten()):
                plt.vlines(0,np.amin(Points_diff[:,dim2,:]), np.amax(Points_diff[:,dim2,:]), alpha=0.5, linestyle = ':')
            if not all(x>=0 for x in Points_diff[:,dim2,:].flatten()):
                plt.hlines(0,np.amin(Points_diff[:,dim1,:]), np.amax(Points_diff[:,dim1,:]), alpha=0.5, linestyle = ':')

            context0, context1 = '',''
            norm0, norm1 = '',''
            if params.plot_norm and (descr[0] in params.dic_norm ): norm0 = ', ' + params.dic_norm[descr[0]]
            if params.plot_norm and (descr[1] in params.dic_norm ): norm1 = ', ' + params.dic_norm[descr[1]]
            if not params.compare_contexts:
                if descr[0] in liste_descrContext:
                    if params.memory_size>=2: context0 = ', '+'Memory: {} chords, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
                    else: context0 = ', '+'Memory: {} chord, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
                if descr[1] in liste_descrContext:
                    if params.memory_size>=2: context1 = ', '+'Memory: {} chords, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
                    else: context1 = ', '+'Memory: {} chord, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
            plt.xlabel(descr[0][0].upper() + descr[0][1:] + suff0 + norm0 + context0)
            plt.ylabel(descr[1][0].upper() + descr[1][1:] + suff1 + norm1 + context1)
            if params.one_track and not params.compare_contexts:
                plt.title(title + ' (' + descr[0][0].upper() + descr[0][1:] + suff0 + ', ' + descr[1][0].upper() + descr[1][1:] + suff1 + ')\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')
            else:
                if params.compare_instruments: goal = 'Timbre comparaison'
                elif params.compare_scores: goal = 'Score comparaison'
                elif params.one_track and params.compare_contexts: goal = 'Context comparaison'
                if type_Normalisation == 'by curve':
                    plt.title(goal + '\n' + title + ' (' + descr[0][0].upper() + descr[0][1:] + suff0 + ', ' + descr[1][0].upper() + descr[1][1:] + suff1 + ')\n' + 'Normalisation curve by curve' + '\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')
                else:
                    plt.title(goal + '\n' + title + ' (' + descr[0][0].upper() + descr[0][1:] + suff0 + ', ' + descr[1][0].upper() + descr[1][1:] + suff1 + ')\n' + 'Normalisation on all the curves ' + '\n' + type_Temporal[0].upper() + type_Temporal[1:] + ' Representation')

        plt.legend(frameon=True, framealpha=0.75)
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, ('7/4','9/4','11/4'), title = 'Mesures :', loc='upper right',frameon=True, framealpha=0.75)
        plt.show()



    elif type == 'temporal':

        # Construction de onset_times régulièrement espacé
        # if onset_times is None:
        #     onset_times = [0.]
        #     for t in range(Points.shape[2]):
        #         onset_times.append(1.+ t*2)
        #     onset_times.append(onset_times[-1] + 2)
        #     onset_times.append(onset_times[-1] + 1)

        # n_frames = len(onset_times)-1
        n_frames = len(onset_times_graph)-1

        if simultaneity:
            m = len(descr)
            sc = 0
            plt.figure(2,figsize=(12.5, 7.5))

            # Plot Score
            if score:
                sc = 1
                img=mpimg.imread(score)
                score = plt.subplot(m+sc,1,1)
                plt.axis('off')
                score.imshow(img)
                # plt.title(title)

            for i, des in enumerate(descr):
                dim = space.index(des)
                ax = plt.subplot(m+sc,1,sc+1+i)
                ax.get_xaxis().set_visible(False)
                plt.xlim(onset_times_graph[0],onset_times_graph[-1])
                norm = ''
                if params.plot_norm and (des in params.dic_norm ): norm = ', ' + params.dic_norm[des]
                # ax.set_title(des[0].upper() + des[1:] + norm)



                plt.vlines(onset_times_graph[1:n_frames], 0.0, np.nanmax(Points[:,dim]), color='k', alpha=0.9, linestyle='--')
                for j, track in enumerate(liste_timbres_or_scores):
                    # Legend
                    context = ''
                    if des in liste_descrContext:
                        if params.compare_contexts:
                            if track[1]>=2: context = '\n' + 'Memory: {} chords, decr = {}'.format(track[1], track[2])
                            else: context = '\n' + 'Memory: {} chord, decr = {}'.format(track[1], track[2])
                        else:
                            if params.memory_size>=2: context = '\n' + 'Memory: {} chords, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
                            else: context = '\n' + 'Memory: {} chord, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)

                    if params.compare_contexts: track_print = ''
                    else: track_print = track# + '\n'

                    # Descripteurs statiques
                    if len(Points[j,dim]) == n_frames-2:
                        #Remplacement des None par les valeurs précédentes
                        for t in range(n_frames-2):
                            if np.isnan(Points[j,dim,t]):
                                Points[j,dim,t] = Points[j,dim,t-1]
                        plt.hlines(Points[j,dim].tolist(),onset_times_graph[1:(n_frames-1)], onset_times_graph[2:n_frames], color=['r','b','g','c','m','y','b','r','g'][j], label=track_print + context)
                        # plt.hlines(Erase(Points[j,dim].tolist(), [e-1 for e in erase]), Erase(onset_times_graph[1:(n_frames-1)], [e-1 for e in erase]), Erase(onset_times_graph[2:n_frames], [e-1 for e in erase]), color=['b','r','g','c','m','y','b','r','g'][j], label=track_print + context)

                    # Descripteurs dynamiques
                    elif len(Points[j,dim]) == n_frames-3:
                        # plt.plot(Erase(onset_times_graph[2:(n_frames-1)], [e-1 for e in erase]), Erase(Points[j,dim].tolist(),[e-1 for e in erase]),['r','b','g','c','m','y','b','r','g'][j]+'o', label= track_print)# + context)
                        # plt.hlines(Erase(Points[j,dim].tolist(), [e-1 for e in erase]), [t-0.50 for t in Erase(onset_times_graph[2:(n_frames-1)], [e-1 for e in erase])], [t+0.50 for t in Erase(onset_times_graph[2:(n_frames-1)], [e-1 for e in erase])], color=['b','r','g','c','m','y','b','r','g'][j], alpha=0.9, linestyle=':'  )
                        plt.plot(onset_times_graph[2:(n_frames-1)], Points[j,dim].tolist(),['r','b','g','c','m','y','b','r','g'][j]+'o', label= track_print)# + context)
                        plt.hlines(Points[j,dim].tolist(), [t-0.50*(j==1)-0.75*(j==0) for t in onset_times_graph[2:(n_frames-1)]], [t+0.50*(j==1)+0.75*(j==0)  for t in onset_times_graph[2:(n_frames-1)]], color=['r','b','g','c','m','y','b','r','g'][j], alpha=0.9, linestyle=':'  )

                handles, labels = ax.get_legend_handles_labels()
                # ax.legend(reversed(handles), ('Smith','Brunn','Yanchenko'), title = des[0].upper() + des[1:] + norm, loc='upper right',frameon=True, framealpha=0.75)
                if i == 0: ax.legend(reversed(handles), ('Original','Modified version'), title = des[0].upper() + des[1:] + norm, loc='upper right',frameon=True, framealpha=0.75)
                else: ax.legend([], (), title = des[0].upper() + des[1:] + norm, loc='upper right',frameon=True, framealpha=0.75)
                # plt.legend(frameon=True, framealpha=0.75)

        else:
            m = len(liste_timbres_or_scores) * len(descr)
            sc = 0
            plt.figure(3,figsize=(13, 7))
            # Plot Score
            if score:
                sc = 1
                img=mpimg.imread(score)
                score = plt.subplot(m+sc,1,1)
                plt.axis('off')
                score.imshow(img)
                plt.title(title)

            for i, des in enumerate(descr):
                dim = space.index(des)
                for j, track in enumerate(liste_timbres_or_scores):
                    if i+j == 0: ax1 = plt.subplot(m+sc,1,sc+1)
                    else: plt.subplot(m+sc,1,i*len(liste_timbres_or_scores)+j+1+sc)


                    plt.vlines(onset_times_graph[1:n_frames], 0.0, max(Points[j,dim]), color='k', alpha=0.9, linestyle='--')

                    # Legend
                    context = ''
                    norm = ''
                    if params.plot_norm and (des in params.dic_norm ): norm = '\n' + params.dic_norm[des]
                    if des in liste_descrContext:
                        if params.compare_contexts:
                            if track[1]>=2: context = '\n' + 'Memory: {} chords, decr = {}'.format(track[1], track[2])
                            else: context = '\n' + 'Memory: {} chord, decr = {}'.format(track[1], track[2])
                        else:
                            if params.memory_size>=2: context = '\n' + 'Memory: {} chords, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)
                            else: context = '\n' + 'Memory: {} chord, decr = {}'.format(params.memory_size, params.memory_decr_ponderation)

                    if params.compare_contexts: track_print = ''
                    else: track_print = track + '\n'

                    # Descripteurs statiques
                    if len(Points[j,dim]) == n_frames-2:
                        plt.hlines(Points[j,dim].tolist(), onset_times_graph[1:(n_frames-1)], onset_times_graph[2:n_frames], color=['b','r','g','c','m','y','b','r','g'][i], label=track_print + des[0].upper() + des[1:] + norm + context)
                    # Descripteurs dynamiques
                    elif len(Points[j,dim]) == n_frames-3:
                        plt.plot(onset_times_graph[2:(n_frames-1)], Points[j,dim],['b','r','g','c','m','y','b','r','g'][i]+'o', label=(track_print + des[0].upper() + des[1:]) + norm + context)
                        plt.hlines(Points[j,dim], [t-0.25 for t in onset_times_graph[2:(n_frames-1)]], [t+0.25 for t in onset_times_graph[2:(n_frames-1)]], color=['b','r','g','c','m','y','b','r','g'][i], alpha=0.9, linestyle=':'  )
                    plt.legend(frameon=True, framealpha=0.75)
                    plt.xlim(onset_times_graph[0],onset_times_graph[-1])

        plt.tight_layout()
        plt.show()






spaceStat_NoCtx = ['roughness', 'harmonicity', 'concordance', 'concordanceTot', 'concordance3']
spaceStat_Ctx = ['harmonicityContext', 'roughnessContext']
spaceStat = spaceStat_NoCtx + spaceStat_Ctx
spaceDyn_NoCtx = ['harmonicChange', 'diffConcordance', 'crossConcordance', 'crossConcordanceTot']
spaceDyn_Ctx = ['harmonicNovelty','diffConcordanceContext', 'diffRoughnessContext']
spaceDyn = spaceDyn_NoCtx + spaceDyn_Ctx


if params.one_track:

    # CHARGEMENT DES SONS ET DE LA PARTITIONs
    title = 'Unison'
    subTitle = 'Unisson'
    instrument = 'Organ'
    if title in params.dic_duration:
        duration = params.dic_duration[title] # en secondes
    else : duration = 100.0 # en secondes
    # DISTRIBUTION
    if title in params.dic_distribution: distribution = params.dic_distribution[title]
    else :distribution = params.distribution


    y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+subTitle+'.wav', duration = duration, sr=None)
    if distribution == 'voix':
        y1, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+subTitle+'-Soprano.wav', duration = duration, sr = None)
        y2, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+subTitle+'-Alto.wav', duration = duration, sr = None)
        y3, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+subTitle+'-Tenor.wav', duration = duration, sr = None)
        y4, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+subTitle+'-Bass.wav', duration = duration, sr = None)
        # y5, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+subTitle+'-Baryton.wav', duration = duration, sr = None)
    elif distribution == 'themeAcc':
        y1, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-theme.wav', duration = duration, sr = None)
        y2, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-acc.wav', duration = duration, sr = None)


    if title in params.dic_noteMin:
        Notemin = params.dic_noteMin[title]
    else: Notemin = 'G3'

    # Notemax = 'C10'
    Notemax = 'B9'
    score = '/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/'+subTitle+'-score.png'
    # if os.path.exists('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/'+title+'-score.png'):
    #     score = '/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/'+subTitle+'-score.png'
    # else: score = []



    # DETECTION D'ONSETS
    delOnsets = []
    addOnsets = []


    # Chargement de onset_frames
    def OpenFrames(title):
        #1 - Avec Sonic Visualiser
        if os.path.exists('Onset_given_'+title+'.txt'):
            onsets = []
            with open('Onset_given_'+title+'.txt','r') as f:
                for line in f:
                    l = line.split()
                    onsets.append(float(l[0]))
            onset_times = np.asarray(onsets)
            onset_frames = librosa.time_to_frames(onset_times, sr=sr, hop_length = STEP)

        #2 - À partir de la partition en musicxml
        elif os.path.exists('Onset_given_'+title+'_score'):
            with open('Onset_given_'+title+'_score', 'rb') as f:
                onsets = pickle.load(f)
                onset_times = np.asarray(onsets)
                onset_frames = librosa.time_to_frames(onset_times, sr=sr, hop_length = STEP)

        #3 - Avec ma méthode de calcul automatique
        elif os.path.exists('Onset_given_'+title):
            with open('Onset_given_'+title, 'rb') as f:
                onset_frames = pickle.load(f)

        #4 - Pas d'onsets préchargés
        else: onset_frames = []
        # for i,time in enumerate(onset_frames)
        return onset_frames

    # onset_frames = OpenFrames('Cadence_M3_NoDelay')
    onset_frames = OpenFrames(subTitle)

    # CRÉATION DE L'INSTANCE DE CLASSE
    if distribution == 'record':
        S = SignalSepare(y, sr, [], Notemin, Notemax,onset_frames, delOnsets, addOnsets, score, instrument)
    elif distribution == 'voix':
        S = SignalSepare(y, sr, [y1,y2,y3,y4], Notemin, Notemax,onset_frames, delOnsets, addOnsets, score, instrument)
    elif distribution == 'themeAcc':
        S = SignalSepare(y, sr, [y1,y2], Notemin, Notemax,onset_frames, delOnsets, addOnsets, score, instrument)

    S.DetectionOnsets()
    # with open('Onset_given_Cadence_M', 'wb') as g:
    #      pickle.dump(S.onset_frames, g)
    S.Clustering()



    if not params.Matrix and not params.compare_contexts:
        # space = ['energy', 'roughness','tension']#['energy','tension','roughness','harmonicity']#['energy','concordance','concordance3','concordanceTot']#
        space = ['tension']
        #'energy','harmonicity', 'roughness', 'concordance'
        S.Context()
        if params.simpl: S.SimplifySpectrum()
        S.ComputeDescripteurs(space = space)
        # print(S.activation[:,1:-1])
        S.Affichage(space = space, end = duration)
        if params.test_stability:
            def stability(power):
                descr = np.divide(getattr(S, space[1])[1:-1], np.power(S.energy[1:-1], power))
                return np.std(descr)/np.mean(descr)

            res = minimize(stability, 1, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
            print('\n    {}, {}\n    Avec normalisation théorique : δ = {}\n    Avec normalisation optimale (α = {}) : δ = {}\n    Rapport theor/opt = {}\n'.format(title, space[1], round(stability(params.dic_test_norm[space[1]]), 3), round(res.x[0], 3),round(stability(res.x[0]),3), round(stability(params.dic_test_norm[space[1]])/stability(res.x[0]), 3)))



    elif params.compare_contexts:
        list_context = [('mean', 1, 1),('mean',2,1),('mean',3,1)]
        dic_context = {list_context[i]:i+1 for i in range(len(list_context))}
        dic_context[('mean', 0, 1)] = 0


        space = ['diffConcordanceContext']
        # Construction Matrice Points
        Points = []
        for ctx in list_context:
            S.Context(type = ctx[0], size = ctx[1]-1, decr = ctx[2])
            S.ComputeDescripteurs(space = space)
            Points.append(S.Points(space))

        Points = np.asarray(Points)
        print(Points.shape)
        # Points = Normalise(Points, list_context, dic_context)
        spacePlot = space
        Visualize(Points, spacePlot, space, list_context, dic_context, type='temporal', simultaneity = True, score = S.score, onset_times = S.onset_times,onset_times_graph = [0,2.5,5,6,6.6,7.3,7.9,9.2,10.8,12.1,12.8,13.5,14.1,14.9,15.5,16.6,17.6,19.3,20])


    elif params.Matrix:
        liste = ['']
        dic = {liste[i]:i+1 for i in range(len(liste))}
        space = spaceDyn_NoCtx

        # Construction_Points(liste, title, space, Notemin, Notemax, dic, filename = 'Points_Beethoven31s_Dyn.npy', name_OpenFrames = 'Beethoven_31s', duration = duration)
        Points = np.load('Points_Beethoven31s_Dyn.npy')
        Points = Normalise(Points, liste, dic)
        spacePlot = ['diffConcordance', 'harmonicChange']
        Visualize(Points, spacePlot, space, liste, dic)



    # Nmin = int(S.sr/(S.fmax*(2**(1/BINS_PER_OCTAVE)-1)))
    # Nmax = int((S.sr/(S.fmin*(2**(1/BINS_PER_OCTAVE)-1))))
    # print(Nmin/S.sr, Nmax/S.sr)

if params.compare:
    title = 'Herzlich1'
    space = ['harmonicChange','diffConcordance','diffRoughness']
    # space = ['concordance','concordance3','concordanceTot','roughness','harmonicity']
    liste_subtitles = ['Herzlich1_1_bis','Herzlich1_1']
    dic_subtitles = {liste_subtitles[i]:i+1 for i in range(len(liste_subtitles))}
    # duration = params.dic_duration[title] # en secondess
    Notemin = params.dic_noteMin[title]
    Notemax = 'D9'


    # Construction_Points(liste_subtitles, title, space, Notemin, Notemax, dic_subtitles, filename = 'Points_Herzlich1_1_comp_Stat.npy', share_onsets = False, liste_no_verticality = [i for i in [4,6,8,12,14,16,18]])#, name_OpenFrames = 'Fratres'), duration = duration)
    Points = np.load('Points_Herzlich1_1_comp_Dyn.npy')

    Points = Normalise(Points, liste_subtitles, dic_subtitles)
    spacePlot = ['harmonicChange']
    # spacePlot = ['roughness', 'harmonicity']

    # Visualize(Points, spacePlot, space, liste_subtitles, dic_subtitles, liste_annot = liste_annot)# liste_Points = [Points1,Points2,Points3])
    score = '/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/'+'Herzlich1_1_comp'+'-score.png'
    # onset_times = np.load('Onset_times_Points_' + title + '.npy')
    Visualize(Points, spacePlot, space, liste_subtitles, dic_subtitles, type = 'temporal', simultaneity = True, score = score, onset_times_graph = params.dic_xcoords['Herzlich1_1'])


    # Points1 = Points[:,:,0:3]
    # Points2 = Points[:,:,4:9]
    # Points3 = Points[:,:,10:17]

if params.compare_instruments:

    liste_timbres = ['Bourdon8', 'Cheminée8', 'Flûte4', 'Flûte2', 'Octave2','Brd + Chm', 'Brd + Fl4', 'Chm + Fl2', 'Chm + Fl4', 'Fl4 + Fl2','Tutti']
    dic_timbres = {liste_timbres[i]:i+1 for i in range(len(liste_timbres))}

    space = spaceStat_NoCtx

    title = 'Cadence_M5'
    duration = 9.0
    score = '/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/'+ 'CadenceM2' +'-score.png'
    Notemin = 'C3'
    Notemax = 'E9'

    # Construction_Points(liste_timbres, title, space, Notemin, Notemax, dic_timbres,filename = 'Points_Timbres_Dyn.npy', name_OpenFrames = 'Cadence_M', duration = duration)

    Points = np.load('Points_Timbres_CadM_Stat.npy')
    liste_timbres = ['Brd + Fl4']
    Points = Normalise(Points, liste_timbres, dic_timbres)
    Disp, Disp_by_descr,Disp_by_time = Dispersion(Points)
    Inertie_tot, Inertie_inter = Inerties(Points)
    descr_sorted, disp_sorted = MinimizeDispersion(Disp_by_descr, space)
    descrs_max_sep = MaximizeSeparation(Inertie_tot, Inertie_inter, space)
    spacePlot = ['harmonicity', 'concordance']
    # spacePlot = descr_sorted[0:2]
    # spacePlot = descrs_max_sep

    Clustered(Points, spacePlot, space)
    Visualize(Points, spacePlot, space, liste_timbres, dic_timbres)#, liste_timbres_or_scores = ['Bourdon8', 'Cheminée8','Brd + Chm','Tutti'])
    # Visualize(Points, spacePlot, space, liste_timbres[:2], dic_timbres, type = 'temporal',score = score)

if params.compare_scores:
    # title = 'Cadence_M'
    title = 'Cadence_M'
    duration = 9.0
    liste_scores = ['Cadence 1', 'Cadence 2','Cadence 3', 'Cadence 4', 'Cadence 5', 'Cadence 6', 'Cadence 7', 'Cadence 8', 'Cadence 9' ]
    # liste_scores = ['Cadence 3', 'Cadence 4','Cadence 5']
    dic_scores = {liste_scores[i]:i+1 for i in range(len(liste_scores))}

    space = spaceStat_NoCtx
    Notemin = 'C2'
    Notemax = 'E9'

    # Construction_Points(liste_scores, title, space, Notemin, Notemax, dic_scores, filename = 'Points_Dispo_Cad_Norm_Stat.npy', name_OpenFrames = 'Cadence_M', duration = duration)

    Points = np.load('Points_Dispo_Cad_Stat.npy')
    # liste_scores = ['Cadence 3','Cadence 4','Cadence 5']
    Points = Normalise(Points, liste_scores, dic_scores)
    Disp, Disp_by_descr,Disp_by_time = Dispersion(Points)
    Inertie_tot, Inertie_inter = Inerties(Points)
    descr_sorted, disp_sorted = MinimizeDispersion(Disp_by_descr, space)
    descrs_max_sep = MaximizeSeparation(Inertie_tot, Inertie_inter, space)
    spacePlot = ['harmonicity', 'concordanceTot']
    # spacePlot = descr_sorted[0:2]
    # spacePlot = descrs_max_sep

    Clustered(Points, spacePlot, space)
    Visualize(Points, spacePlot, space, liste_scores, dic_scores)



























    # S.Sort(space = space)
    #
    # with open('nouv4', 'wb') as g:
    #     pickle.dump(S.harmonicNovelty, g)
    #
    # Nmin = int(S.sr/(S.fmax*(2**(1/BINS_PER_OCTAVE)-1)))
    # Nmax = int((S.sr/(S.fmin*(2**(1/BINS_PER_OCTAVE)-1))))
