import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


Notemin = 'G2'
Notemax = 'D9'
BINS_PER_OCTAVE = 12*8
STEP = 512
FILTER_SCALE = 1
fmin = librosa.note_to_hz(Notemin)
fmax = librosa.note_to_hz(Notemax)
cmap='gray_r'
spectral_reloc = True
hpss = True
margin = 20

n_bins = int((librosa.note_to_midi(Notemax) - librosa.note_to_midi(Notemin))*BINS_PER_OCTAVE/12)
WINDOW = np.hanning

# y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/Cadence_M3.wav', duration = 9)
y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/TimbreComparaison/Fichiers son/1NoteLent.wav', duration = 96)
Chrom = np.abs(librosa.cqt(y, sr=sr, hop_length = STEP, fmin=fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=n_bins, window=WINDOW, filter_scale = FILTER_SCALE))
Nt = len(Chrom[0])


# Relocalisation
if spectral_reloc:
    freq_analyse = [fmin*2**(k/BINS_PER_OCTAVE) for k in range(n_bins)]
    N = [round(sr * FILTER_SCALE/(f*(2**(1/BINS_PER_OCTAVE)-1))) for f in freq_analyse]
    N_sample = [round(n/STEP) for n in N]
    Chrom_copy = np.copy(Chrom)
    for k in range(n_bins):
        for n in reversed(range(Nt)):
            if n <= N_sample[k]: Chrom[k,n] = Chrom_copy[k,n]
            else: Chrom[k,n] = Chrom_copy[k,n-int(N_sample[k]/2)]

# Decompo hpss
if hpss:
    Chrom = librosa.decompose.hpss(Chrom, margin=margin)[0]
    # Chrom_harm = librosa.decompose.hpss(Chrom, margin=margin)[0]
    # Chrom_percu = librosa.decompose.hpss(Chrom, margin=margin)[1]
    # Chrom_noise = Chrom - Chrom_harm - Chrom_percu




######################################
# Visualisation du spectre
fig= plt.figure(figsize=(13, 7))
img = librosa.display.specshow(librosa.amplitude_to_db(Chrom, ref=np.max),bins_per_octave=BINS_PER_OCTAVE, fmin=fmin, sr=sr, x_axis='time', y_axis='cqt_note', cmap = cmap)
plt.title('Constant-Q power spectrum')
plt.axis('tight')
plt.tight_layout()
plt.show()

######################################
# # Décomposition HPSS
# plt.figure(2,figsize=(9, 7.5))
# plt.subplot(2, 1, 1)
# librosa.display.specshow(librosa.amplitude_to_db(Chrom_harm,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
# plt.title('Partie harmonique (β = {})'.format(margin))
#
# plt.subplot(2, 1, 2)
# librosa.display.specshow(librosa.amplitude_to_db(Chrom_percu + Chrom_noise,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
# plt.title('Partie percussive + bruit (β = {})'.format(margin))
#
# # plt.subplot(3, 1, 3)
# # librosa.display.specshow(librosa.amplitude_to_db(Chrom_noise,ref=np.max), bins_per_octave=BINS_PER_OCTAVE, fmin=fmin, y_axis='cqt_note', x_axis='time',cmap=cmap)
# # plt.title('Résidu (bruit)')
#
# plt.tight_layout()
# plt.show()
