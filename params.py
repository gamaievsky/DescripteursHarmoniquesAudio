#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

one_track = True
Matrix = False
test_stability = False
compare_instruments = False
compare_scores = False

type_Temporal = 'differential' #'static', 'differential'
type_Normalisation = 'by timbre' #'by curve', 'by timbre'
visualize_time_grouping = True
visualize_trajectories = False

correlation = False
pca = False

# Distribution des pistes :
distribution = 'voix' # 'voix', 'themeAcc', 'record'


#general load WAV
begin = 0.0 #start reading the analysed signal after this time (in seconds)s
duration = None #60*2 # duration of the analysed signal (in seconds) ou None.
begin_ref = 0.0 #beginning of the reference signal for computing cosine distance between clusters (in seconds).
end_ref = 0.05 #end of the reference signal for computing cosine distance between clusters (in seconds).

#Analyse
BINS_PER_OCTAVE_ONSETS = 12*2
BINS_PER_OCTAVE = 12*8
WINDOW = np.hanning
FILTER_SCALE = 1
#Relocalisation
spectral_reloc = True
#Decomposition hpss
decompo_hpss = True
margin = 3

dic_duration = {'EssaiNuances' : 49.0,'EssaiNuances2' : 49.0 , 'EssaiNuances4' : 49.0, 'EssaiUnisson' : 26.0, 'EssaiUnissonInterv' : 20.0, 'Schnittke_Cor_Lent' : 368.0, 'Schnittke_Cor': 23.0, 'Schnittke_Implor': 48, 'Beethoven': 62}

#Matrice d'activation (quelles pistes sont à prendre en compte à quel moment ?)
list_calcul_nombre_notes = ['EssaiUnisson', 'EssaiUnissonInterv']
seuil_activation = 0.01

#NFFT = 2 ** 11 #(> 2**10) duration of analysis window in samples for feature extraction only.
#STEP = NFFT / 2 #(>2**6) et (STEP < NFFT) 50% overlap between time windows / also sub-frequency after analyzing spectral structure.


## DETECTION ONSETS
if True:
    SemiManual = False

    # Parametres de la fonction de seuil pour la detection d'onset_strength
    α = 130
    ω = 1
    H =  30
    #Filtre sur les onsets
    T = 0.3 #(en secondes)
    T_att = 0.0

    #(α,β,H,T,T_att)
    paramsDetOnsets_Palestrina = [172, 1, 30, 0.3, 0.1]
    paramsDetOnsets_PalestrinaM = [105, 1, 30, 0.3, 0.1]
    paramsDetOnsets_SuiteAccords = [83, 1, 40, 0.3, 0.1]
    paramsDetOnsets_AccordsParalleles = [140, 1, 30, 0.3, 0.1]
    paramsDetOnsets_SuiteAccordsPiano = [140, 1, 30, 0.3, 0.1]
    paramsDetOnsets_4Notes = [140, 1, 30, 0.3, 0.1]
    paramsDetOnsets_Schubert = [140, 1, 30, 0.3, 0.1]
    paramsDetOnsets_Purcell = [140, 1, 30, 0.3, 0.1]
    paramsDetOnsets_SchubertRecord = [140, 1, 30, 0.3, 0.1]


    #Ajustements = ([frames], [times])
    [delOnsets_Palestrina, addOnsets_Palestrina] = [[], []]
    [delOnsets_PalestrinaM, addOnsets_PalestrinaM] = [[6,8,9], [4.5]] #7,8,9 juste pour éviter la division du même accord
    [delOnsets_SuiteAccords, addOnsets_SuiteAccords] = [[2,3,4,6,8,10,11,12],[6.1]]
    delOnsets_AccordsParalleles, addOnsets_AccordsParalleles = [5,6],[6.05,7.5]#6.05
    delOnsets_SuiteAccordsPiano, addOnsets_SuiteAccordsPiano = [],[7.5]
    delOnsets_4Notes, addOnsets_4Notes= [],[22]
    delOnsets_Schubert, addOnsets_Schubert= 'all',[i + 0.1 for i in [0,2,3,4,6,7,8,10,11,12,14,15,16,18,19,20,22,23,24,26,28,30]]
    delOnsets_Purcell, addOnsets_Purcell= 'all',[i + 3.1 for i in [0,1.5,3,4.5,6,7.5,9,9.75,10.5,12,13.5,15,16.5,18,19.5,21,21.75,22.5,23.63,24,25.5,27,28.5,30,30.75,31.5,33,34.5,36,37.5,38.25,39,40.5,42,46.5]]
    delOnsets_SchubertRecord, addOnsets_SchubertRecord= 'all',[i + 0.1 for i in [0,2,3,4,6,7,8,10,11,12,14,15,16,18,19,20,22,23,24,26,28,30]]

    #delOnsets_AccordsParalleles, addOnsets_AccordsParalleles = [4,5,6],[5.0,6.05,7.5]#Dévié

    #Tri des fréquences qui entrent en compte dans le calcul de la déviation
    triFreq = False





## NORMALISATIONS
norm_conc = True  # True : normalisation par l'énergie
norm_conc3 = True # True : normalisation par la norme 3
norm_concTot = True # True : normalisation par la norme N, où N est le nombre de notes
spectrRug_Simpl = False
type_rug = 'produit' #'produit', 'minimum'
norm_rug = True #True : normalisation par l'énergie si type_rug = 'produit', et la norme 1 sinon
norm_crossConc = True  #False : chrom_crossConcordance porte déjà la normalisation de la concordance, True : norm par la concordance
norm_crossConcTot = True #False : chrom_crossConcordance porte déjà la normalisation de la concordanceTot, True : norm par la concordanceTot
type_harmChange = 'absolute' # 'absolute', 'relative'
norm_harmChange = 'general' # 'None', 'frame_by_frame', 'general'
norm_diffConc = 'None' # 'note_by_note', 'general', 'None'
norm_harmonicity = 2 # La puissance dans le calcul de l'harmonicité. 1 : amplitude, 2 : énergie

memory_size = 1 # "full", int # entier n>=1, auquel cas la mémoire ne dure que n+1 accords
memory_type = 'mean' #'max','mean'
memory_decr_ponderation = 1
norm_Novelty = True #True : normalisation par l'énergie
type_Novelty = 'dyn' #'dyn', 'stat'
norm_rugCtx = True
norm_diffConcCtx = 'general' #'general', 'None'

dic_test_norm = {'roughness': 1, 'roughnessSignal': 1, 'concordance' : 1, 'harmonicity' : 0}
Norm = {'concordance':norm_conc, 'concordance3':norm_conc3, 'roughness':norm_rug, 'roughnessSignal':norm_rug, 'concordanceTot':norm_concTot, 'harmonicChange':norm_harmChange, 'diffConcordance':norm_diffConc, 'crossConcordance':norm_crossConc, 'crossConcordanceTot':norm_crossConcTot}
dic_norm = {}
for descr in ['concordance', 'concordance3', 'concordanceTot', 'roughness', 'roughnessSignal','crossConcordance','crossConcordanceTot']:
    if Norm[descr]: dic_norm[descr] = 'Normalised'
    else: dic_norm[descr] = 'Not normalised'
for descr in ['harmonicChange', 'diffConcordance']:
    if Norm[descr] == 'None': dic_norm[descr] = 'Not normalised'
    else : dic_norm[descr]='Normalisation : {}'.format(Norm[descr])

norm_nbNotes = True


#PLOT
plot_onsets = False
plot_pistes = False
plot_partiels = False
plot_context = False
plot_decompo_hpss = False
plot_chromDescr = False
plot_descr = True
plot_OneDescr = False
plot_abstr = False
plot_symb = False
plot_sorted = False
plot_compParam = False
sorted_reverse = False
play = False

plot_norm = True
plot_score = True
cmap='gray_r' # 'magma','gray_r','coolwarm'

color_abstr = 'b'
color_abstr_numbers = 'black'
link_abstr = True



#PARAMETRES DES DESCRIPTEURS
#Roughness
β1 = 3.5
β2 = 5.75
P_ref = 20*(10**(-6))
#Tension:
δ = 0.6

#Harmonicity:
σ = 12 * 4 # en divisions de l'octave
κ = 15
decr = 0
