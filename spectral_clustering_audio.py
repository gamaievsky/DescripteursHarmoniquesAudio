#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
======================
Laplacian segmentation
======================

This notebook implements the laplacian segmentation method of
`McFee and Ellis, 2014 <http://bmcfee.github.io/papers/ismir2014_spectral.pdf>`_,
with a couple of minor stability improvements.
This implementation is available at https://librosa.github.io/librosa/auto_examples/plot_segmentation.html

Additional functions have been added to the core segmentation:
 - unsupervised determination of the number of clusters suitable for the running task
 - different feature packages: spectral, cepstral and chroma.
 - a cosine distance between the different clusters that is plot together with cluster segmentation
 - a set of parameters reported in params.py file necessary for tuning the segmentation model.

usage:
python3 spectral_clustering_audio.py audiofilename.wav [.mp3]

Input:
 - name of audio file to be analyzed

Output:
 - Segmentation and grouping of the different musical sections synchronized on user-chosen onsets
 - Optional plots of similarity and recurrence matrix
 - Optional timestamps text file with parameters and time boundaries
"""

# Code source by Marie Tahon (2018) adapted from Brian McFee (2014)
# License: ISC


###################################
# Imports
#   - numpy for basic functionality
#   - scipy for graph Laplacian
#   - matplotlib for visualization
#   - sklearn.cluster for K-Means, for metrics and scaling.
#   - warnings to delete warning message for scipy package



from __future__ import division
import numpy as np
import scipy
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import sys, os
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec

import sklearn.cluster
from sklearn.preprocessing import scale
import sklearn.metrics
import sklearn.utils


import librosa
import librosa.display

import cluster_rotate
import params

plt.rcParams.update({'font.size': 8})

BINS_PER_OCTAVE = params.BINS_PER_OCTAVE
N_OCTAVES = params.N_OCTAVES
NFFT = int(params.NFFT)
STEP = int(params.STEP)



#######################################
def detect_onsets(y, sr, M):
	#detect onsets
	oenv = librosa.onset.onset_strength(S=M, sr=sr)
	# Detect events without backtracking
	onset_raw = librosa.onset.onset_detect(onset_envelope=oenv, backtrack=False)
	## Backtrack the events using the onset envelope
	onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)
	# we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
	onset_frames = librosa.util.fix_frames(onset_raw, x_min=0, x_max=M.shape[1]-1)
	onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length = STEP)
	# To reduce dimensionality, we'll beat-synchronous the CQT
	Msync = librosa.util.sync(M, onset_raw, aggregate=np.median)

	if params.onset_plot:
		plt.figure(figsize=(12, 4))
		plt.plot(oenv, label='Onset strength')
		plt.vlines(onset_raw, 0, oenv.max(), label='Raw onsets')
		plt.vlines(onset_bt, 0, oenv.max(), label='Backtracked', color='r')
		plt.legend(frameon=True, framealpha=0.75)
		plt.tight_layout()

		plt.figure(figsize=(12, 4))
		plt.subplot(2,1,1)
		plt.title('CQT spectrogram')
		librosa.display.specshow(M, y_axis='cqt_hz', sr=sr, hop_length= STEP, bins_per_octave=BINS_PER_OCTAVE, x_axis='time')
		plt.tight_layout()

		plt.subplot(2,1,2)
		plt.title('CQT spectrogram synchronized on onsets')
		librosa.display.specshow(Msync, bins_per_octave=BINS_PER_OCTAVE, y_axis='cqt_hz', x_axis='time', x_coords=onset_times)
		plt.tight_layout()

	return onset_raw, onset_times, Msync



##############################################
def detect_beats(y, sr, M):
	tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length = STEP, trim=False)
	print('Detected tempo:  {0:.2f} bpm'.format(tempo))
	beat_period = np.diff(librosa.frames_to_time(beats, sr=sr, hop_length= STEP))
	print('mean beat period: {0:.2f} ; std beat period: {1:.2f}'.format(60/np.mean(beat_period), np.std(beat_period)))

	beats_frames = librosa.util.fix_frames(beats, x_min=0, x_max=M.shape[1]-1)
	beat_times = librosa.frames_to_time(beats_frames, sr=sr, hop_length = STEP)

	Msync = librosa.util.sync(M, beats_frames, aggregate=np.median)
	if params.onset_plot:
		plt.figure(figsize=(12, 4))
		plt.subplot(2,1,1)
		plt.title('CQT spectrogram')
		librosa.display.specshow(M, y_axis='cqt_hz', sr=sr, hop_length=STEP, bins_per_octave=BINS_PER_OCTAVE, x_axis='time')
		plt.tight_layout()
		# For plotting purposes, we'll need the timing of the beats
		# we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)

		plt.subplot(2,1,2)
		plt.title('CQT spectrogram synchronized on beats')
		librosa.display.specshow(Msync, bins_per_octave=BINS_PER_OCTAVE, y_axis='cqt_hz', x_axis='time', x_coords=beat_times)
		plt.tight_layout()
	return beats_frames, beat_times, Msync

##############################################
def no_onsets(sr, M):

	onsets = np.arange(0, M.shape[1])
	onset_times = librosa.samples_to_time(onsets, sr=sr/STEP)

	if params.onset_plot:
		plt.figure(figsize=(12, 4))
		plt.title('CQT spectrogram')
		librosa.display.specshow(M, y_axis='cqt_hz', sr=sr, bins_per_octave=BINS_PER_OCTAVE, x_axis='time', x_coords=onset_times)
		plt.tight_layout()

	return onsets, onset_times, M


def get_manual_beats(sr, M, filename):
	with open(filename, 'r') as f:
		data = f.readlines()
	times = np.array([float(x.strip()) for x in data[1:]])
	frames = np.array([int(x * sr / STEP) for x in times])
	onsets = librosa.util.fix_frames(frames, x_min=0, x_max=M.shape[1]-1)
	onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length = STEP)

	Msync = librosa.util.sync(M, onsets, aggregate=np.median)

	if params.onset_plot:
		plt.figure(figsize=(12, 4))
		plt.subplot(2,1,1)
		plt.title('CQT spectrogram')
		librosa.display.specshow(M, y_axis='cqt_hz', sr=sr, hop_length=STEP, bins_per_octave=BINS_PER_OCTAVE, x_axis='time')
		plt.tight_layout()

		plt.subplot(2,1,2)
		plt.title('CQT spectrogram synchronized on beats')
		librosa.display.specshow(Msync, bins_per_octave=BINS_PER_OCTAVE, y_axis='cqt_hz', x_axis='time', x_coords=onset_times)
		plt.tight_layout()

	return onsets, onset_times, Msync


def extract_onsets(y, sr, manual_opt):
	method = params.onset
	#compute the CQT transform C: np.array((252, Tmax*sr/STEP))
	C = librosa.amplitude_to_db(librosa.core.magphase(librosa.cqt(y=y, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE, hop_length = STEP))[0], ref=np.max)
	#to reduce dimensionality, we'll onset-synchronous the CQT
	#onset is a vector of onset indexes np.array((N+1,)) including 0
	#onset_times is a vector of onset times np.array((N+1,)) including 0
	#Csync is the CQT transform synchronized on onsets np.array((252, N))
	if method == 'no':
		onset, onset_times, Csync = no_onsets(sr, C)
	elif method == 'onset':
		onset, onset_times, Csync = detect_onsets(y, sr, C)
	elif method == 'beat':
		onset, onset_times, Csync = detect_beats(y, sr, C)
	elif method == 'manual':
		onset, onset_times, Csync = get_manual_beats(sr, C, manual_opt)
	else:
		print('onset parameter is not well-defined')
		sys.exit()

	return onset, onset_times, Csync



def build_weighted_rec_matrix(M):
	# Let's build a weighted recurrence affinity matrix using onset-synchronous CQT

	# the similarity matrix is filtered to prevent linkage errors and fill the gaps
	# the filter corresponds to a width=3 time window and a majority vote.
	R = librosa.segment.recurrence_matrix(M, width=3, mode='affinity',sym=True)

	# Enhance diagonals with a median filter
	df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
	Rf = df(R, size=(1, 7))
	return Rf



def build_seq_matrix(M, x):
	#build the sequence matrix using feature-similarity
	#Rpath[i, i+/-1] = \exp(- |M[i] - C[i+/-1]|^2 / sigma^2)`

	#synchronize features with onsets
	Msync = librosa.util.sync(M, x, aggregate=np.median)
	#Msync = M #pas de syncrhonisation

	#normalize (rescale) features between 0 and 1
	Msync_normed = scale(Msync)

	#constant scaling
	path_distance = np.sum(np.diff(Msync_normed, axis=1)**2, axis=0)
	#sigma is the median distance between successive beats/onsets.
	sigma = np.median(path_distance)
	path_sim = np.exp(-path_distance / sigma)

	#local scaling from A Spectral Clustering Approach to Speaker Diarization, Huazhong Ning, Ming Liu, Hao Tang, Thomas Huang
	R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
	return R_path


def build_laplacian_and_evec(Rf, R_path, opt, onsets):

	# And compute the balanced combination A of the two similarity matrices Rf and R_path
	deg_path = np.sum(R_path, axis=1)
	deg_rec = np.sum(Rf, axis=1)
	mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)
	print('Optimal weight value (mu): {0:.2f}'.format(mu))

	A = mu * Rf + (1 - mu) * R_path

	# Plot the resulting graphs
	if opt: plot_similarity(Rf, R_path, A, onsets)

	# L: symetrized normalized Laplacian
	L = scipy.sparse.csgraph.laplacian(A, normed=True)

	# and its spectral decomposition (Find eigenvalues w and optionally eigenvectors v of matrix L)
	evals, evecs = np.linalg.eigh(L)
	print('L shape:', L.shape)

	# We can clean this up further with a median filter.
	# This can help smooth over small discontinuities
	evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

	# cumulative normalization is needed for symmetric normalize laplacian eigenvectors
	Cnorm = np.cumsum(evecs**2, axis=1)**0.5

	return Cnorm, evals, evecs





################################################
def compute_nb_clusters(method, evals, evecs, Tmax):

	if method == 'fixed':
		c = params.cluster_nb # list
	elif method == 'max':
		nc = []
		for it in range(params.cluster_max):
			nc.append(cluster_rotate.cluster_rotate(evecs/Cnorm, evals, range(1,10), 1, False))
		c = [int(np.mean(nc))+1]
	elif method == 'evals':
		ind = np.where(1- evals > 0.75)[0]
		#print(ind)
		return [len(ind)+1 ]
	elif method in ['silhouette', 'davies_bouldin', 'calinski_harabaz']:
		list_k = range(2,50,2)
		Cnorm = np.cumsum(e**2, axis=1)**0.5 #eigenvectors in input
		for k in list_k:
			print('nb of clusters:', k)
			X = e[:, :k] / Cnorm[:, k-1:k]
			# Let's use these k components to cluster beats into segments
			# (Algorithm 1)
			KM = sklearn.cluster.KMeans(n_clusters=k)
			seg_ids = KM.fit_predict(X)
			score = []
			if method == 'silhouette':
				score.append(sklearn.metrics.silhouette_score(X, seg_ids, metric='euclidean')) #max (proche de 1)
			elif method == 'davies_bouldin':
				score.append(davies_bouldin_score(X, seg_ids)) #min
			elif method == 'calinski_harabaz':
				score.append(sklearn.metrics.calinski_harabaz_score(X, seg_ids)) #max

		if method == 'silhouette':
			return list_k[np.argmax(score)]
		elif method == 'davies_bouldin':
			return list_k[np.argmin(score)]
		elif method == 'calinski_harabaz':
			return list_k[np.argmax(score)]

	else:
		print('method for finding the right number of clusters is unknown')
		sys.exit()

	print('nb of clusters:', c)

	return c



def davies_bouldin_score(X, labels):
	"""Computes the Davies-Bouldin score.
	The score is defined as the ratio of within-cluster distances to
	between-cluster distances.
	Read more in the :ref:`User Guide <davies-bouldin_index>`.
	Parameters
	----------
	X : array-like, shape (``n_samples``, ``n_features``)
	List of ``n_features``-dimensional data points. Each row corresponds
	to a single data point.
	labels : array-like, shape (``n_samples``,)
	Predicted labels for each sample.
	Returns
	-------
	score: float
	The resulting Davies-Bouldin score.
	References
	----------
	.. [1] `Davies, David L.; Bouldin, Donald W. (1979).
	"A Cluster Separation Measure". IEEE Transactions on
	Pattern Analysis and Machine Intelligence. PAMI-1 (2): 224-227`_
	"""
	X, labels = sklearn.utils.check_X_y(X, labels)
	le = sklearn.preprocessing.LabelEncoder()
	labels = le.fit_transform(labels)
	n_samples, _ = X.shape
	n_labels = len(le.classes_)
	if not 1 < n_labels < n_samples:
		raise ValueError("Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)" % n_labels)

	intra_dists = np.zeros(n_labels)
	centroids = np.zeros((n_labels, len(X[0])), dtype=np.float)
	for k in range(n_labels):
		cluster_k = sklearn.utils.safe_indexing(X, labels == k)
		centroid = cluster_k.mean(axis=0)
		centroids[k] = centroid
		intra_dists[k] = np.average(sklearn.metrics.pairwise.pairwise_distances(cluster_k, [centroid]))

	centroid_distances = sklearn.metrics.pairwise.pairwise_distances(centroids)


	if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
		return 0.0

	score = (intra_dists[:, None] + intra_dists) / centroid_distances
	score[score == np.inf] = np.nan

	return np.mean(np.nanmax(score, axis=1))



def plot_similarity(Rf, R_path, A, onset_times):

	plt.figure(figsize=(12, 4))
	plt.subplot(1, 3, 1)
	librosa.display.specshow(Rf, cmap='inferno_r', y_axis='time', y_coords=onset_times)
	plt.title('Long-range recurrence similarity (Rrec)')
	plt.subplot(1, 3, 2)
	librosa.display.specshow(R_path, cmap='inferno_r')
	plt.title('Local path similarity (Rloc)')
	plt.subplot(1, 3, 3)
	librosa.display.specshow(A, cmap='inferno_r')
	plt.title('Combined graph (A = m Rrec + (1-m) Rloc)')
	plt.tight_layout()



def plot_structure(Rf, X, seg_ids, k, onset_times):

	fig_s = plt.figure(figsize=(12, 4))
	colors = plt.get_cmap('Paired', k)

	ax_s1 = fig_s.add_subplot(1, 3, 2)
	librosa.display.specshow(Rf, cmap='inferno_r')
	ax_s1.set_title('Long-range recurrence similarity (Rrec)')
	ax_s2 =fig_s.add_subplot(1, 3, 1)
	librosa.display.specshow(X, y_axis='time', y_coords=onset_times)
	ax_s2.set_title('Structure components (Eigen vectors)')
	ax_s3 = fig_s.add_subplot(1, 3, 3)
	librosa.display.specshow(np.atleast_2d(seg_ids).T, cmap=colors)
	ax_s3.set_title('Estimated segments')
	plt.colorbar(ticks=range(k))
	plt.tight_layout()


#################################################
def compute_musical_density(C, onset_times, w, alpha):
	N = C.shape[1]
	density = []
	for n in range(N):
		t1 = np.min([onset_times[-1], onset_times[n] + w])
		t2 = np.min([onset_times[-1] -w, onset_times[n]])
		idw = np.where((onset_times < t1) & (onset_times >= t2))
		#if n + w < :
		threshold_chroma = np.max(C[:,idw])
		#else:
			#threshold_chroma = np.mean(C[:, N - w : N])
		idx = np.where(C[:,n] > alpha * threshold_chroma)
		density.append(len(idx[0]))

	return density



def plot_features(X, onsets, onset_times):

	Xsync = librosa.util.sync(X, onsets, aggregate=np.median)

	#print(X.shape, Xsync.shape)
	#print(onset_times)

	if params.feat[0] == 'chroma':
		fig_c = plt.figure(figsize=(12, 6))
		ax0_c = fig_c.add_subplot(3,1,1)
		ax0_c.set_title('onset-synchronous chroma (12)')
		#ax0_c.pcolor(distance, cmap = 'plasma')
		librosa.display.specshow(Xsync[:12,:], y_axis='chroma', x_axis='time', x_coords=onset_times, cmap = 'OrRd')
		#plt.colorbar()

		ax1_c = fig_c.add_subplot(3,1,2, sharex = ax0_c)
		ax1_c.set_title('onset-synchronous delta chroma (12)')
		librosa.display.specshow(np.abs(Xsync[12:,:]), y_axis='chroma', x_axis='time', x_coords=onset_times, cmap = 'OrRd')
		#plt.colorbar()

		density = compute_musical_density(Xsync[:12,:], onset_times, params.norm_density_win, params.alpha)
		print(len(onset_times), len(density))
		ax2_c = fig_c.add_subplot(3,1,3, sharex = ax0_c)
		ax2_c.set_title('musical density')
		ax2_c.plot(onset_times, density)
		plt.tight_layout()

	elif params.feat[0] == 'cepstral':
		fig_s = plt.figure(figsize=(12, 6))
		ax0_s = fig_s.add_subplot(3,1,1)
		ax0_s.set_title('onset-synchronous MFCC (20)')
		librosa.display.specshow(Xsync[:21,:], x_axis='time', x_coords=onset_times)
		#plt.colorbar()
		#plt.tight_layout()

		ax1_s = fig_s.add_subplot(3,1,2, sharex = ax0_s)
		ax1_s.set_title('onset-synchronous delta MFCC (20)')
		librosa.display.specshow(np.abs(Xsync[20:,:]), x_axis='time', x_coords=onset_times)
		#plt.colorbar()

		density = compute_musical_density(Xsync[:21,:], onset_times, params.norm_density_win, params.alpha)
		ax2_s = fig_s.add_subplot(3,1,2, sharex = ax0_s)
		ax2_s.set_title('musical density')
		ax2_s.plot(onset_times, density)
		plt.tight_layout()
	else:
		print('these parameters can not be plot')



def load_wav_percu(filename, start, duration, opt_percussive_part):
	y, sr = librosa.load(filename, offset=start, duration = duration)

	if opt_percussive_part:
	#separate harmonics and percussives into two wavforms
		y_harmo, yo = librosa.effects.hpss(y)
		librosa.output.write_wav(filename + '_harmo.wav', y_harmo, sr)
		librosa.output.write_wav(filename + '_percu.wav', y_percu, sr)
		return yo, sr
	else:
		return y, sr






################################################
def feature_extraction(y, sr, opt_tuning):

	if opt_tuning:
		#extraction of tuning
		A440 = librosa.estimate_tuning(y=y, sr=sr, resolution=1e-3)
		print('Deviation from A440 is : {0:.2f}'.format(A440))
	else:
		A440 = 0.0

	print('Features for local similarity: ', ' '.join(params.feat))
	full = []
	idx_chroma = 0

	if 'cepstral' in params.feat:
		mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 20, n_fft = NFFT, hop_length = STEP)
		mfcc_delta = librosa.feature.delta(mfcc)
		fcep = np.concatenate((mfcc, mfcc_delta), axis=0)
		full.append(fcep)

	if 'chroma' in params.feat:
		chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma = 12, n_octaves = N_OCTAVES, hop_length = STEP, norm = None, tuning= A440)
		chroma_delta = librosa.feature.delta(chroma)
		fchr = np.concatenate((chroma, chroma_delta), axis=0)
		idx_chroma = len(full)
		full.append(fchr)

	if 'spectral' in params.feat:
		centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft = NFFT, hop_length = STEP)
		contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft = NFFT, n_bands=6, hop_length = STEP)
		flatness = librosa.feature.spectral_flatness(y=y, n_fft = NFFT, hop_length = STEP)
		rolloff05 = librosa.feature.spectral_rolloff(y=y, sr= sr, n_fft = NFFT, hop_length = STEP, roll_percent= 0.05)
		rolloff25 = librosa.feature.spectral_rolloff(y=y, sr= sr, n_fft = NFFT, hop_length = STEP, roll_percent= 0.25)
		rolloff50 = librosa.feature.spectral_rolloff(y=y, sr= sr, n_fft = NFFT, hop_length = STEP, roll_percent= 0.50)
		rolloff75 = librosa.feature.spectral_rolloff(y=y, sr= sr, n_fft = NFFT, hop_length = STEP, roll_percent= 0.75)
		rolloff95 = librosa.feature.spectral_rolloff(y=y, sr= sr, n_fft = NFFT, hop_length = STEP, roll_percent= 0.95)
		spec = np.concatenate((centroid, contrast, flatness, rolloff05,rolloff25,rolloff50,rolloff75,rolloff95), axis=0)
		spec_delta = librosa.feature.delta(spec)
		fspec = np.concatenate((spec, spec_delta), axis = 0)
		full.append(fspec)

	full = np.array(full)[0]

	print('feature shape', full.shape)
	return full, idx_chroma

def extract_time_boundaries(cluster_ids, onsets, nb_frames, sr):

	# Locate segment boundaries from the label sequence
	bound_beats = 1 + np.flatnonzero(cluster_ids[:-1] != cluster_ids[1:])

	# Count beat 0 as a boundary
	bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

	# Compute the segment label for each boundary
	bound_labels = list(cluster_ids[bound_beats])

	# Convert beat indices to frames
	bound_frames = onsets[bound_beats]

	# Make sure we cover to the end of the track
	bound_frames = librosa.util.fix_frames(bound_frames,  x_min=None, x_max=nb_frames-1)
	bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length = STEP)

	return bound_times, bound_labels



##################################
def extract_cosine_distance_clusters(center_clusters, distance_ref, type_dist = 'cos'):
	distance = []

	for center in center_clusters:
		if type_dist == 'cos':
			distance.append( scipy.spatial.distance.cosine( center, distance_ref) )
		elif type_dist == 'eucl':
			distance.append(np.sqrt( np.sum( (center - distance_ref)**2) ))

	return distance


def extract_distance_between_clusters(center_clusters, type_dist = 'cos'):
	distance = np.zeros((center_clusters.shape))

	for i, center_i in enumerate(center_clusters):
		for j, center_j in enumerate(center_clusters):
			if type_dist == 'cos':
				distance[i,j] = scipy.spatial.distance.cosine( center_i, center_j)
			elif type_dist == 'eucl':
				distance[i,j] = np.sqrt( np.sum( (center_i - center_j)**2) )

	x = range(i+1)
	y = range(j+1)
	xloc = [c + 0.5 for c in x]
	cx = [str(c) for c in x]
	#print(cx)
	fig_d, ax_d = plt.subplots(figsize=(5, 4))
	p_d = ax_d.pcolor(distance, cmap = 'inferno_r')
	cb = fig_d.colorbar(p_d)
	ax_d.xaxis.set_ticks(xloc)
	ax_d.xaxis.set_ticklabels(cx)
	ax_d.yaxis.set_ticks(xloc)
	ax_d.yaxis.set_ticklabels(cx)
	ax_d.set_title('Distance between clusters')
	ax_d.set_xlabel('clusters numbers')
	plt.tight_layout()

	return distance



def extract_ref_signal(X, onset_times):
	ind = np.where((onset_times >= params.begin_ref) & (onset_times < params.end_ref))
	return X[ind,:]



def main():


	parser = argparse.ArgumentParser(description='Segmentation and clustering of musical sections with spectral clustering (Laplacian matrix and eigen values)')
	parser.add_argument('filename', type=str, help='name of audio file')
	parser.add_argument('manual_onset', nargs='?', type=str, help='name of the file containing manual annotations for onset timestamps (with method=manual)')

	args = parser.parse_args()

	#==================
	# Signal processing
	#==================

	#extract waveform from audio signal of given duration and begining. If onset_percu is True, extract only percussive part of the signal.
	y, sr = load_wav_percu(args.filename, params.begin, params.duration, params.onset_percu)
	print('signal shape:', y.shape, ' sr=', sr, 'win duration=%.2f' %(NFFT / sr))

	#extract acoustic feature from audio signal feat is a matrix np.array((nb features, Tmax*sr/STEP))
	feat, idx_chroma = feature_extraction(y, sr, params.opt_tuning)

	#extract onset indexes and times + onset-synchronous CQT transform on onsets.
	onsets, onset_times, Csync = extract_onsets(y, sr, args.manual_onset)

	#if 'chroma' in params.feat:
	#	compute_musical_density(Csync, onset_times, idx_chroma, params.norm_density_win, params.alpha, sr)

	if params.plot_features: plot_features(feat, onsets, onset_times)

	#================
	# Affinity matrix
	#================

	#compute a non-negative affinity matrix using onset-synchronous CQT (with Gaussian kernel)
	#represent local consistency of timbral (CQT) features
	Rf = build_weighted_rec_matrix(Csync)

	#compute a non-negative affinity matrix using onset-synchronous feature matrix (with Gaussian kernel)
	#represent long-range repeating forms of harmonic features
	R_path = build_seq_matrix(feat, onsets)

	#compute Laplacian (sequence augmented affinity matrix) as a linear combination of Rf and Rpath and extract eigenvalues and vectors.
	Cnorm, evals, evecs = build_laplacian_and_evec(Rf, R_path, params.plot_simi, onset_times)


	#===========
	# Clustering
	#===========

	#determine number of clusters kl is a list of potential numbers of cluster.
	kl = compute_nb_clusters(params.cluster_method, evals, evecs, y.shape[0]*sr)
	N_CLUST = len(kl)


	#=================
	# Start plotting
	#=================
	import matplotlib.patches as patches
	fig_f = plt.figure(figsize = (12, 3+2*N_CLUST))
	#fig.subplots_adjust(hspace=.5)

	#plot onset-synchronous CQT
	hr = [1] * (N_CLUST +1)
	hr[0] = 2
	gs = gridspec.GridSpec(1 + N_CLUST,1, height_ratios=hr)
	ax_f0 = fig_f.add_subplot(gs[0])
	librosa.display.specshow(Csync, y_axis='cqt_hz', sr=sr, hop_length = STEP, bins_per_octave=BINS_PER_OCTAVE, x_axis='time', x_coords=onset_times)
	#librosa.display.specshow(feat, y_axis='chroma', x_axis='time') #ou
	ax_f0.set_title('CQT spectrogram synchronized {0}'.format(params.onset))

	for it, k in enumerate(kl):
		#limit the number of clusters per second
		if k > params.cluster_nb_max*sr*y.shape[0]:
			k = params.cluster_nb_max*sr*y.shape[0]
		print('nb of clusters: {} for it {}/{}'.format(k, it, N_CLUST))

		#for k clusters, use the first k normalized eigenvectors.
		#X can be interpretable as an onset-synchronous matrix containing relevant feature information for local and log-range structure segmentation
		X = evecs[:, :k] / Cnorm[:, k-1:k]

		#onsets are grouped into k clusters, each cluster having its own acoustic characteristics
		KM = sklearn.cluster.KMeans(n_clusters=k)
		#seg_ids is a np.array((label)) label being a number corresponding to one cluster seg_ids[i] is the label of onset i
		seg_ids = KM.fit_predict(X)

		#if needed compute the cosine distance between each cluster and a reference taken at the very begining of th signal
		#KM.cluster_centers_ : array, [n_clusters, n_features]
		if params.cluster_dist:
			ref_signal = extract_ref_signal(X, onset_times)
			distance_cosine_cluster = extract_cosine_distance_clusters( KM.cluster_centers_, np.mean(X[:10*NFFT,:], axis=0))
		else:
			distance_cosine_cluster = None

		if params.plot_dist:
			distance_between_clusters = extract_distance_between_clusters( KM.cluster_centers_ )


		# and plot the resulting structure representation
		if params.plot_struct: plot_structure(Rf, X, seg_ids, k, onset_times)

		bound_times, bound_labels = extract_time_boundaries(seg_ids, onsets, feat.shape[1], sr)
		freqs = librosa.cqt_frequencies(n_bins=Csync.shape[0], fmin=librosa.note_to_hz('C1'), bins_per_octave=BINS_PER_OCTAVE)

		timestamps_name = os.path.splitext(args.filename)[0] + '_timestamps.txt'

		#=============
		# Plot results
		#=============


		cmap = plt.get_cmap('Paired', k)
		#write header of text file with parameters.
		if params.timestamps:
			f = open(timestamps_name, 'a')
			f.write('WIN = {0:.2f} sec, NFFT = {1}, STEP = {2}, begin = {3}, duration = {4}\n'.format(NFFT / sr, NFFT, STEP, params.begin, params.duration))
			f.write('Nb of clusters: {0} obtained with method {1} and features {2}\n'.format(k, params.cluster_method, '-'.join(params.feat)))

		#plot onset-synchronous CQT
		#if it == 0:


		#plot segmentation and clusters grouping (+ cosine distance.)
		#also write obtained boundaries in the text file.
		ax_f1 = fig_f.add_subplot(gs[it + 1], sharex = ax_f0)
		for interval, label in zip(zip(bound_times, bound_times[1:]), bound_labels):
			if params.timestamps: f.write('{0:.2f} \t {1:.2f} \t {2} \n'.format(interval[0], interval[1], label))
			if params.cluster_dist: ax_f1.plot([interval[0], interval[1]],[distance_cosine_cluster[label], distance_cosine_cluster[label]], 'k')
			ax_f1.add_patch(patches.Rectangle((interval[0], 0), interval[1] - interval[0], 1, facecolor=cmap(label), alpha=1))
			ax_f1.text(interval[0]+(interval[1]-interval[0])/2, 0.9, label, fontsize=8)
		if params.timestamps: f.close()

		#plt.subplots_adjust(hspace=.0)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()



title = 'Palestrina'
# Palestrina, AccordsMajeurs, AccordsMineur, Majeur3et4notes, Majeur3et4notes, Accords3Notes, DispoMajeurMineur, Tension
# Cadence3V, Cadence4VMaj, Cadence4Vmin,
audio = load('/Users/manuel/Dropbox (TMG)/Thèse/code/DescripteursHarmoniquesAudio/'+title+'.wav')
main(audio)
