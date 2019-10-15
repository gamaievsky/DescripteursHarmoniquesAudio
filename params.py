import numpy as np

#general load WAV
begin = 0.0 #start reading the analysed signal after this time (in seconds)
duration = None #60*2 # duration of the analysed signal (in seconds) ou None.
begin_ref = 0.0 #beginning of the reference signal for computing cosine distance between clusters (in seconds).
end_ref = 0.05 #end of the reference signal for computing cosine distance between clusters (in seconds).


BINS_PER_OCTAVE_ONSETS = 12*2
BINS_PER_OCTAVE = 12*8
WINDOW = np.hanning
FILTER_SCALE = 1
decompo_hpss = True

#NFFT = 2 ** 11 #(> 2**10) duration of analysis window in samples for feature extraction only.
#STEP = NFFT / 2 #(>2**6) et (STEP < NFFT) 50% overlap between time windows / also sub-frequency after analyzing spectral structure.


## DETECTION ONSETS

SemiManual = False
#Paramètres  de la fonction de seuil pour la détection d'onset_strength
α = 140
β = 1
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
#Ajustements = ([frames], [times])
[delOnsets_Palestrina, addOnsets_Palestrina] = [[], []]
[delOnsets_PalestrinaM, addOnsets_PalestrinaM] = [[6,8,9], [4.5]] #7,8,9 juste pour éviter la division du même accord
[delOnsets_SuiteAccords, addOnsets_SuiteAccords] = [[2,3,4,6,8,10,11,12],[6.1]]
delOnsets_AccordsParalleles, addOnsets_AccordsParalleles = [5,6],[6.05,7.5]#6.05
delOnsets_SuiteAccordsPiano, addOnsets_SuiteAccordsPiano = [],[7.5]
#delOnsets_AccordsParalleles, addOnsets_AccordsParalleles = [4,5,6],[5.0,6.05,7.5]#Dévié

#Tri des fréquences qui entrent en compte dans le calcul de la déviation
triFreq = True


## NORMALISATIONS
norm_conc = 'piste_by_piste' # 'None' 'piste_by_piste', 'energy_total'
norm_diss = True
norm_crossConc = 'energy' #'energy', 'energy + conc' ##energy : même normalisation que dans le calcul de la consonance
norm_crossConcTot = 'energy' #'energy', 'energy + conc' ##energy : même normalisation que dans le calcul de la consonance totale
type_harmChange = 'relative' # 'absolute', 'relative'
norm_harmChange = 'general' # 'None', 'frame_by_frame', 'general'
norm_diffConc = 'piste_by_piste' # 'piste_by_piste', 'frame-by-frame'
norm_harm = 2 # La puissance dans le calcul de l'harmonicité. 1 : amplitude, 2 : énergie


#PLOT
plot_onsets = True
plot_pistes = False
plot_decompo_hpss = False
plot_chromDescr = False
plot_descr = True
plot_score = False
plot_symb = False
play = False


#PARAMETRES DES DESCRIPTEURS
#Dissonance
S0 = 0.24
S1 = 0.021
S2 = 19
B1 = 3.5
B2 = 5.75

#Tension:
δ = 0.6

#Harmonicity:
σ = 12 * 4 # en divisions de l'octave
κ = 20
decr = 0
