#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

#general load WAV
begin = 0.0 #start reading the analysed signal after this time (in seconds)
duration = None #60*2 # duration of the analysed signal (in seconds) ou None.
begin_ref = 0.0 #beginning of the reference signal for computing cosine distance between clusters (in seconds).
end_ref = 0.05 #end of the reference signal for computing cosine distance between clusters (in seconds).

#Analyse
BINS_PER_OCTAVE_ONSETS = 12*2
BINS_PER_OCTAVE = 12*8
WINDOW = np.hanning
FILTER_SCALE = 1
#Decomposition hpss
decompo_hpss = True
margin = 3

#NFFT = 2 ** 11 #(> 2**10) duration of analysis window in samples for feature extraction only.
#STEP = NFFT / 2 #(>2**6) et (STEP < NFFT) 50% overlap between time windows / also sub-frequency after analyzing spectral structure.


## DETECTION ONSETS

SemiManual = False

# Parametres de la fonction de seuil pour la detection d'onset_strength
α = 140
ω = 1
H =  30
#Filtre sur les onsets
T = 0.3 #(en secondes)
T_att = 0.5

#(α,β,H,T,T_att)
paramsDetOnsets_Palestrina = [172, 1, 30, 0.3, 0.1]
paramsDetOnsets_PalestrinaM = [105, 1, 30, 0.3, 0.1]
paramsDetOnsets_SuiteAccords = [83, 1, 40, 0.3, 0.1]
paramsDetOnsets_AccordsParalleles = [140, 1, 30, 0.3, 0.1]
paramsDetOnsets_SuiteAccordsPiano = [140, 1, 30, 0.3, 0.1]
paramsDetOnsets_Septiemes = [140, 1, 30, 0.3, 0.1]
#Ajustements = ([frames], [times])
[delOnsets_Palestrina, addOnsets_Palestrina] = [[], []]
[delOnsets_PalestrinaM, addOnsets_PalestrinaM] = [[6,8,9], [4.5]] #7,8,9 juste pour éviter la division du même accord
[delOnsets_SuiteAccords, addOnsets_SuiteAccords] = [[2,3,4,6,8,10,11,12],[6.1]]
delOnsets_AccordsParalleles, addOnsets_AccordsParalleles = [5,6],[6.05,7.5]#6.05
delOnsets_SuiteAccordsPiano, addOnsets_SuiteAccordsPiano = [],[7.5]
delOnsets_Septiemes, addOnsets_Septiemes= [],[9]
#delOnsets_AccordsParalleles, addOnsets_AccordsParalleles = [4,5,6],[5.0,6.05,7.5]#Dévié

#Tri des fréquences qui entrent en compte dans le calcul de la déviation
triFreq = True

#Matrice d'activation (quelles pistes sont à prendre en compte à quel moment ?)
seuil_activation = 0.01

## NORMALISATIONS
norm_conc = 'note_by_note' # 'None' 'note_by_note', 'chord_by_chord'
norm_conc3 = 'norme3' # 'None' 'energy', 'norme3'
spectrDiss_Simpl = False
type_diss = 'produit' #'produit', 'minimum'
norm_diss = True
norm_crossConc = 'energy + conc' #'energy', 'energy + conc' ##energy : même normalisation que dans le calcul de la consonance
norm_crossConcTot = 'energy + concTot' #'energy', 'energy + concTot' ##energy : même normalisation que dans le calcul de la consonance totale
type_harmChange = 'absolute' # 'absolute', 'relative'
norm_harmChange = 'general' # 'None', 'frame_by_frame', 'general'
norm_diffConc = 'note_by_note' # 'note_by_note', 'chord_by_chord', ('frame-by-frame')
norm_harm = 2 # La puissance dans le calcul de l'harmonicité. 1 : amplitude, 2 : énergie


#PLOT
plot_onsets = False
plot_pistes = False
plot_partiels = False
plot_decompo_hpss = True
plot_chromDescr = False
plot_descr = False
plot_OneDescr = False
plot_abstr = False
plot_symb = False
play = False

plot_score = True
cmap='magma' # 'magma','gray_r','coolwarm'

color_abstr = 'b'
link_abstr = True


#PARAMETRES DES DESCRIPTEURS
#Dissonance
β1 = 3.5
β2 = 5.75
P_ref = 20*(10**(-6))
#Tension:
δ = 0.6

#Harmonicity:
σ = 12 * 4 # en divisions de l'octave
κ = 15
decr = 0

print(α)
