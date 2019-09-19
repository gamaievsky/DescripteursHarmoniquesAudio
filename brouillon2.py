# Decompose a magnitude spectrogram into 32 components with NMF

y, sr = librosa.load(librosa.util.example_audio_file())
S = np.abs(librosa.stft(y))
comps, acts = librosa.decompose.decompose(S, n_components=8)
comps
# array([[  1.876e-01,   5.559e-02, ...,   1.687e-01,   4.907e-02],
# [  3.148e-01,   1.719e-01, ...,   2.314e-01,   9.493e-02],
# ...,
# [  1.561e-07,   8.564e-08, ...,   7.167e-08,   4.997e-08],
# [  1.531e-07,   7.880e-08, ...,   5.632e-08,   4.028e-08]])
acts
# array([[  4.197e-05,   8.512e-03, ...,   3.056e-05,   9.159e-06],
# [  9.568e-06,   1.718e-02, ...,   3.322e-05,   7.869e-06],
# ...,
# [  5.982e-05,   1.311e-02, ...,  -0.000e+00,   6.323e-06],
# [  3.782e-05,   7.056e-03, ...,   3.290e-05,  -0.000e+00]])

# Sort components by ascending peak frequency

comps, acts = librosa.decompose.decompose(S, n_components=16,
                                          sort=True)

# Or with sparse dictionary learning

import sklearn.decomposition
T = sklearn.decomposition.MiniBatchDictionaryLearning(n_components=16)
scomps, sacts = librosa.decompose.decompose(S, transformer=T, sort=True)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S,
                                                 ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Input spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.subplot(3, 2, 3)
librosa.display.specshow(librosa.amplitude_to_db(comps,
                                                 ref=np.max),
                         y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Components')
plt.subplot(3, 2, 4)
librosa.display.specshow(acts, x_axis='time')
plt.ylabel('Components')
plt.title('Activations')
plt.colorbar()
plt.subplot(3, 1, 3)
S_approx = comps.dot(acts)
librosa.display.specshow(librosa.amplitude_to_db(S_approx,
                                                 ref=np.max),
                         y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Reconstructed spectrogram')
plt.tight_layout()
plt.show()
