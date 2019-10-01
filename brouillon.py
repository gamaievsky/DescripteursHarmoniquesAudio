from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from operator import truediv

import librosa
import librosa.display
import IPython.display as ipd

import pyaudio
import wave
import sys
import soundfile as sf
import os


if os.path.exists("stereo_file.wav"):
  os.remove("stereo_file.wav")
else:
  print("The file does not exist")


y, sr = librosa.load("PalestrinaM.wav")
audio = sf.write('stereo_file.wav', y, sr)


class AudioFile:
    chunk = 1024

    def __init__(self, file):
        """ Init audio stream """
        self.wf = wave.open(file, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(self.chunk)
        while data != '':
            self.stream.write(data)
            data = self.wf.readframes(self.chunk)

    def close(self):
        """ Graceful shutdown """
        self.stream.close()
        self.p.terminate()

# Usage example for pyaudio
a = AudioFile("Palestrina.wav")
a.play()
a.close()







CHUNK = 1024

if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

wf = wave.open('/Users/manuel/Github/DescripteursHarmoniquesAudio/Exemples/Palestrina.wav', 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# read data
data = wf.readframes(CHUNK)

# play stream (3)
while len(data) > 0:
    stream.write(data)
    data = wf.readframes(CHUNK)

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio
p.terminate()











if (2<=4<5): pass


l = [9,2,1,4,5,6]
print(min(l))

x = np.arange(-2,2,0.01)
a = 0.6
y = np.exp(-(x/a)**2)

plt.figure()
plt.plot(x,y)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

k = 2*np.pi
w = 2*np.pi
dt = 0.01

xmin = 0
xmax = 3
nbx = 100

x = np.linspace(xmin, xmax, nbx)

fig = plt.figure() # initialise la figure
line, = plt.plot([],[])
plt.xlim(xmin, xmax)
plt.ylim(-1,1)

# fonction à définir quand blit=True
# crée l'arrière de l'animation qui sera présent sur chaque image
def init():
    line.set_data([],[])
    return line,

def animate(i):
    t = i * dt
    y = np.cos(k*x - w*t)
    line.set_data(x, y)
    return line,

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, blit=False, interval=20, repeat=False)

plt.show()





import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots()
xdata, ydata = [], []
xdescr, ydescr = [1,2,3,4,5,6,7,8], [1,2,3,4,5,4,3,2]
ln, = plt.plot([], [], 'r'+'--'+'o')

def init():
    ax.set_xlim(0, max(xdescr)+1)
    ax.set_ylim(0, max(ydescr)+1)
    return ln,

def update(frame):
    xdata.append(xdescr[frame])
    ydata.append(ydescr[frame])
    ln.set_data(xdata, ydata)
    ax.annotate(frame+1, (xdescr[frame], ydescr[frame]))
    return ln,


ani = FuncAnimation(fig, update, frames=range(1,len(xdescr)),
                    init_func=init, blit=False, interval=200.3, repeat=False)
plt.show()



import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax1 = plt.subplots(1,1)

def animate(i,argu):
    print(i, argu)

    #graph_data = open('example.txt','r').read()
    graph_data = "1, 1 \n 2, 4 \n 3, 9 \n 4, 16 \n"
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y)+np.sin(2.*np.pi*i/10))
        ax1.clear()
        ax1.plot(xs, ys)
        plt.grid()

ani = animation.FuncAnimation(fig, animate, fargs=[5],interval = 100)
plt.show()
