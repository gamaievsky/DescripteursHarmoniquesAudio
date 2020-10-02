#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

one_track = True
Matrix = False
test_stability = False
compare_contexts = False
compare_instruments = False
compare_scores = False

type_Temporal = 'static' #'static', 'differential'
type_Normalisation = 'by timbre' #'by curve', 'by timbre'
visualize_time_grouping = True
visualize_trajectories = True

correlation = False
pca = False

# Distribution des pistes :
dic_distribution = {'EssaiNuances' : 'record','EssaiNuances2' : 'themeAcc' , 'EssaiNuances4' : 'voix', 'EssaiUnisson' : 'voix', 'EssaiUnissonInterv' : 'voix', 'Schnittke_Cor_Lent' : 'themeAcc', 'Schnittke_Cor': 'themeAcc', 'Schnittke_Cor_T2': 'themeAcc', 'Schnittke_Implor': 'themeAcc', 'Schnittke_Implor_T2': 'themeAcc','Schnittke_Implor_T2_Lent': 'themeAcc', 'Beethoven': 'voix','EssaiNuancesIndiv':'voix', '3Accords':'voix', 'Cadence_M3':'record', 'Crucis':'record', 'Tons':'record', '1Note':'record', 'Tenue':'record','Nuances':'voix','NuanceEnch':'voix','NuancesInd':'voix','Unisson':'voix','Cad':'voix','Fratres':'voix'}
distribution = 'voix' # 'voix', 'themeAcc', 'record'

# Xcoords pour fonction Affichage
dic_xcoords =  {'3Accords':[0,2.3,6.3,10.6,14.2,14.6],'Crucis':[0.0,2.6,5.1,8,10.5,13.3,15.5,18.5], 'Tons':[0.0,2.1,2.9,3.7,4.5,5.3,6.1,6.9,7.7,8.9,9.7,10.5,11.7,12.8,13.6,14.2,15,16,16.8,17.5,18.3,19.6], '1Note':[0,2.9,7,9.5,10.2,10.9,11.5,13,14.7], 'Tenue':[0,2.1,4.5,7.2,9.4,11.3,12.2], 'Nuances':[0,1.9,4,6.1,8.2,10.4,12.5,14.7,16.8,18.6,19],'NuancesEnch':[0,2.5,5.5,8.3,11.3,14.1,17.1,19.4,20.3],'NuancesInd':[0,2.1,5.7,9.4,13,16.7,20,20.5], 'Unisson':[0,1.8,6.2,10.7,15.1,19.1,19.5], 'Cad':[0,2.6,6.0,9.4,12.9,16.4,19.5,20.3],'Schnittke_Cor_T2':[0,2.5,5,6,6.6,7.3,7.9,9.2,10.8,12.1,12.8,13.5,14.1,14.9,15.5,16.6,17.6,19.3,20], 'FratresCourt':[0,2.3,4.3,5.5,6.8,10,12,13.3,14.6,15.9,17.1,19.5,20], 'Fratres':[0,1.5,2.7,3.4,4.1,6.1,7.3,8,8.8,9.5,10.3,12.4,13.6,14.4,15.2,15.9,16.7,17.4,18.2,19.7,20]} #, 'Tenue':[0,3.4,7.3,11.7,15.3,18.5,19.3] [0,1.8,6.2,10.7,15.1,19.1,19.5]

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
margin = 20
#Simplification du spectre
simpl = True
δ = 4 #int, en subdivisions spectrales

dic_duration = {'EssaiNuances' : 49.0,'EssaiNuances2' : 49.0 , 'EssaiNuances4' : 49.0, 'EssaiUnisson' : 26.0, 'EssaiUnissonInterv' : 20.0, 'Schnittke_Cor_Lent' : 368.0, 'Schnittke_Cor': 23.0, 'Schnittke_Cor_T2': 23.0, 'Schnittke_Implor': 48.0, 'Schnittke_Implor_T2': 48.0, 'Schnittke_Implor_T2_Lent': 96.0, 'Beethoven': 62.0,'EssaiNuancesIndiv':26.0, '3Accords':21.0, '3Accords_aigu':21.0,'Cadence_M3':9.0,'Crucis':11,'Tons':39.0,'1Note':9.0, 'Tenue':9.8,'Nuances':33.0,'NuancesEnch':25.0,'NuancesInd':21.0,'Unisson':25.0,'Cad':21.0,'Fratres':41.0}
dic_noteMin = {'EssaiNuances' : 'D4','EssaiNuances2' : 'D4' , 'EssaiNuances4' : 'G3', 'EssaiUnisson' : 'D4', 'EssaiUnissonInterv' : 'A3', 'Schnittke_Cor_Lent' : 'A1', 'Schnittke_Cor': 'A1','Schnittke_Cor_T2': 'A1', 'Schnittke_Implor': 'A1', 'Schnittke_Implor_T2': 'A1', 'Schnittke_Implor_T2_Lent': 'A1', 'Beethoven': 'A1','EssaiNuancesIndiv':'A3','Cadence_M3':'G2','Crucis':'C2','Tons':'C2','1Note':'A2', 'Tenue':'G2','Nuances':'G3','NuancesEnch':'A2','NuancesInd':'G3', 'Unisson': 'A3','Cad':'C3', 'Fratres':'F3'}
#Matrice d'activation (quelles pistes sont à prendre en compte à quel moment ?)
list_calcul_nombre_notes = ['EssaiUnisson', 'EssaiUnissonInterv','Schnittke_Cor','Schnittke_Cor_Lent','Schnittke_Cor_T2','Unisson']
list_3_voix = ['EssaiNuancesIndiv']
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
type_rug_signal = False
rug_simpl = True
mod_rough = 'sethares' #'sethares', 'sethares + KK'
norm_rug = False #True : normalisation par l'énergie si type_rug = 'produit', et la norme 1 sinon
norm_tension = True
norm_harm = True
norm_crossConc = False  #False : chrom_crossConcordance porte déjà la normalisation de la concordance, True : norm par la concordance
norm_crossConcTot = False #False : chrom_crossConcordance porte déjà la normalisation de la concordanceTot, True : norm par la concordanceTot
theme_diffConc = False
norm_diffConc = True
norm_harmonicity = 1 # La puissance dans le calcul de l'harmonicité. 1 : amplitude, 2 : énergie

memory_size = 1 # "full", int # entier n>=1, auquel cas la mémoire ne dure que n accords
memory_type = 'mean' #'max','mean'
memory_decr_ponderation = 1
norm_Novelty = 'general' # 'None', 'frame_by_frame', 'general'
type_Novelty = 'dyn' #'dyn', 'stat'
type_harmChange = 'absolute' # 'absolute', 'relative'
norm_harmChange = 'general' # 'None', 'frame_by_frame', 'general'
norm_rugCtx = True
theme_diffConcCtx = True
norm_diffConcCtx = True # True : Normalisation par l'énergie des deux accords
theme_diffRugCtx = False
norm_diffRugCtx = True

dic_test_norm = {'roughness': 1, 'roughnessSignal': 1, 'concordance' : 1, 'harmonicity' : 0}
Norm = {'concordance':norm_conc, 'concordance3':norm_conc3, 'roughness':norm_rug, 'harmonicity':norm_harm, 'tension':norm_tension, 'roughnessSignal':norm_rug, 'concordanceTot':norm_concTot, 'harmonicChange':norm_harmChange, 'harmonicNovelty':norm_Novelty,'diffConcordance':norm_diffConc, 'diffConcordanceContext':norm_diffConcCtx,'crossConcordance':norm_crossConc, 'crossConcordanceTot':norm_crossConcTot, 'roughnessContext':norm_rugCtx, 'diffRoughnessContext':norm_diffRugCtx}
Theme = {'diffConcordance':theme_diffConc, 'diffConcordanceContext':theme_diffConcCtx, 'diffRoughnessContext':theme_diffRugCtx}
dic_norm = {}
for descr in ['concordance', 'concordance3', 'concordanceTot', 'roughness', 'harmonicity','tension', 'roughnessSignal','roughnessContext','crossConcordance','crossConcordanceTot','diffConcordance','diffConcordanceContext', 'diffRoughnessContext']:
    type = ''
    if (descr in Theme) and Theme[descr]:
        type = ', Theme'
    if Norm[descr]: dic_norm[descr] = 'Normalised' + type
    else: dic_norm[descr] = 'Not normalised' + type

for descr in ['harmonicChange', 'harmonicNovelty']:
    if Norm[descr] == 'None': dic_norm[descr] = 'Not normalised'
    else : dic_norm[descr]='Normalised'
    # else : dic_norm[descr]='Normalisation : {}'.format(Norm[descr])

norm_nbNotes = True


#PLOT
plot_onsets = False
plot_pistes = False
plot_simple = False
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
β2 = 5.0
P_ref = 20*(10**(-6))
#Tension:
δ_tension = 0.6

#Harmonicity:
σ = 12 * 4 # en divisions de l'octave
κ = 15
decr = 0
