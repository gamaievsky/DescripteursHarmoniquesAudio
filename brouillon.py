import librosa
import librosa.display

title = 'CadenceM2_T1'
y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'.wav')
y1, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-Basse.wav')
y2, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-Alto.wav')
y3, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-Soprano.wav')
y4, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/'+title+'-Tenor.wav')
