import pickle
import numpy as np

from music21 import *

title = 'Herzlich1_1_bis'
score = converter.parse('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/'+title+'.musicxml')
Tree = tree.fromStream.asTimespans(score, flatten=True,classList=(note.Note, chord.Chord))

offsets = []
tempo_noire = 20.0
delay = 0.03

for verticality in Tree.iterateVerticalities():
    # Test si la verticalité est en fait le prolongement lié de la précédente, auquel cas on ne tient pas compte de la verticalité
    tied = True
    for elt in verticality.startTimespans:
        if (elt.element.tie == None) or (elt.element.tie.type == 'start'):
            tied = False
            break
    # Ajout des offsets des nouvelles verticalités
    if not tied:
        offsets.append(verticality.offset * 60.0/tempo_noire + delay)

    # Ajout du dernier offset
    if verticality.nextStartOffset == None:
        offsets.append(verticality.bassTimespan.endTime * 60.0/tempo_noire)

# print(offsets)

with open('Onset_given_'+title+'_score', 'wb') as fp:
# with open('Onset_given_Cadeence_M3_NoDelay_score', 'wb') as fp:
    pickle.dump(offsets, fp)
