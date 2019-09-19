#!/usr/bin/python3

# Code source by Marie Tahon (2018)
# from original idea of Jean-Marc Chouvel
# cf http://www.ems-network.org/spip.php?article294
# License: ISC

from __future__ import division
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys, os

import librosa
import librosa.display

import argparse
import params

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")



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
	
	return beats_frames, beat_times, Msync

##############################################
def no_onsets(sr, M):
	
	onsets = np.arange(0, M.shape[1])
	onset_times = librosa.samples_to_time(onsets, sr=sr/STEP)
	
	return onsets, onset_times, M


def get_manual_beats(sr, M, filename):
	with open(filename, 'r') as f:
		data = f.readlines()
	times = np.array([float(x.strip()) for x in data[1:]])
	frames = np.array([int(x * sr / STEP) for x in times])
	onsets = librosa.util.fix_frames(frames, x_min=0, x_max=M.shape[1]-1)
	onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length = STEP)
	
	Msync = librosa.util.sync(M, onsets, aggregate=np.median)

	return onsets, onset_times, Msync


def extract_onsets(y, sr, C, manual_opt):
	method = params.onset
	#compute the CQT transform C: np.array((252, Tmax*sr/STEP))
	#C = librosa.amplitude_to_db(librosa.core.magphase(librosa.cqt(y=y, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE, hop_length = STEP))[0], ref=np.max)
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



def plot_spectrograms(D, Dd, sr):
	fig_s = plt.figure(figsize=(12,4))
	ax_s0 = fig_s.add_subplot(2,1,1)
	librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), y_axis='log', sr=sr, hop_length = STEP, x_axis='time')
	ax_s0.set_title('original CQT spectrogram')
	ax_s1 = fig_s.add_subplot(2,1,2)
	librosa.display.specshow(librosa.amplitude_to_db(Dd, ref=np.max), y_axis='log', sr=sr, hop_length = STEP, x_axis='time')
	ax_s1.set_title('Differential CQT spectrogram')
	plt.tight_layout()

def plot_wavforms(t, y, td, yd):
	plt.figure()
	plt.plot(td, yd, 'r')
	plt.plot(t, y, alpha=0.5)
	plt.xlabel('time (sec.)')
	plt.ylabel('amplitude (UA)')
	plt.legend(('differential TF', 'original'), loc='upper right')
	plt.title('Differential TF')
	plt.tight_layout()


def main():


	parser = argparse.ArgumentParser(description='Computation and visualisation of differential transform based on synchronous beats.')
	parser.add_argument('filename', type=str, help='name of audio file')
	parser.add_argument('manual_onset', nargs='?', type=str, help='name of the file containing manual annotations for onset timestamps (with method=manual)')

	args = parser.parse_args()
	
	y, sr = librosa.load(args.filename, offset=params.begin, duration = params.duration)
	
	## Calculation of Fourrier Transform
	D_mag, D_phase = librosa.core.magphase(librosa.stft(y, n_fft = NFFT, hop_length = STEP))
	
	##synchronisation on onsets
	onset_ech, onset_times, Dsync = extract_onsets(y, sr, D_mag, args.manual_onset)
	
	## Calculation of differential Fourrier Transform
	Dsync_delta = librosa.feature.delta(Dsync)
	
	## back synchronisation on samples.
	D_delta = np.zeros((D_mag.shape[0], D_mag.shape[1]))
	it = 0
	for n in range(D_mag.shape[1]):
		if n in onset_ech:
			D_delta[:,n] = Dsync_delta[:,it]
			it = it + 1
	
	plot_spectrograms(D_mag, D_delta, sr)
	
	D_time = librosa.istft( (D_delta * D_phase), hop_length = STEP)
	print(D_mag.shape, D_delta.shape, Dsync.shape, Dsync_delta.shape, D_time.shape, D_phase.shape)
	
	t = np.linspace(0, y.shape[0]*sr, y.shape[0])
	t_delta = np.linspace(0, y.shape[0]*sr, D_time.shape[0])
	plot_wavforms(t, y, t_delta, D_time)
	plt.show()
	
	
	

if __name__ == '__main__':
	main()

