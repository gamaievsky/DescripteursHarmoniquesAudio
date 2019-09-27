import numpy as np

#general load WAV
begin = 0.0 #start reading the analysed signal after this time (in seconds)
duration = None #60*2 # duration of the analysed signal (in seconds) ou None.
begin_ref = 0.0 #beginning of the reference signal for computing cosine distance between clusters (in seconds).
end_ref = 0.05 #end of the reference signal for computing cosine distance between clusters (in seconds).


BINS_PER_OCTAVE = 12*2
N_OCTAVES = 7
NFFT = 2 ** 11 #(> 2**10) duration of analysis window in samples for feature extraction only.
STEP = NFFT / 2 #(>2**6) et (STEP < NFFT) 50% overlap between time windows / also sub-frequency after analyzing spectral structure.
WINDOW = np.hanning

## DETECTION ONSETS

SemiManual = True
#Paramètres  de la fonction de seuil pour la détection d'onset_strength
α = 83
β = 1
H =  40
#Filtre sur les onsets
T = 0.3 #(en secondes)
T_att = 0.1

#(α,β,H,T,T_att)
paramsDetOnsets_Palestrina = [172, 1, 30, 0.3, 0.1]
paramsDetOnsets_PalestrinaM = [105, 1, 30, 0.3, 0.1]
paramsDetOnsets_SuiteAccords = [83, 1, 40, 0.3, 0.3]
#Ajustements = ([frames], [times])
[delOnsets_Palestrina, addOnsets_Palestrina] = [[], []]
[delOnsets_PalestrinaM, addOnsets_PalestrinaM] = [[6,8,9], [4.5]] #7,8,9 juste pour éviter la division du même accord
[delOnsets_SuiteAccords, addOnsets_SuiteAccords] = [[2,3,4,6,8,10,11,12],[6.1]]

#Tri des fréquences qui entrent en compte dans le calcul de la déviation
triFreq = True


## NORMALISATIONS
norm_spectre = False
norm_conc = True
norm_concTot = True

#PLOT
plot_onsets = False
plot_pistes = False
plot_chromDescr = False
plot_descr = True

#PARAMETRES DES DESCRIPTEURS
#Dissonance
S0 = 0.24
S1 = 0.021
S2 = 19
B1 = 3.5
B2 = 5.75

#Tension:
δ = 0.6
