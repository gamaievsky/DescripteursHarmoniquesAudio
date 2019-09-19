import numpy as np

#general load WAV
begin = 0.0 #start reading the analysed signal after this time (in seconds)
duration = None #60*2 # duration of the analysed signal (in seconds) ou None.
begin_ref = 0.0 #beginning of the reference signal for computing cosine distance between clusters (in seconds).
end_ref = 0.05 #end of the reference signal for computing cosine distance between clusters (in seconds).

## DETECTION ONSETS
BINS_PER_OCTAVE = 12*2
N_OCTAVES = 7
NFFT = 2 ** 11 #(> 2**10) duration of analysis window in samples for feature extraction only.
STEP = NFFT / 2 #(>2**6) et (STEP < NFFT) 50% overlap between time windows / also sub-frequency after analyzing spectral structure.
WINDOW = np.hamming

#Paramètres  de la fonction de seuil pour la détection d'onset_strength
ALPHA = 180
BETA = 1
H =  30

#Filtre sur les onsets
T = 0.25 #(en secondes)
T_att = 0

#Tri des fréquences qui entrent en compte dans le calcul de  la déviation
triFreq = True

#Plot Onsets
plot_onsets = True

#Normalisation
norm_spectre = False
