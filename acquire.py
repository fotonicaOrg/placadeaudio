# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:30:56 2018

@author: Guillermo
"""

import pyaudio
import wave
import sys
import numpy
import matplotlib.pylab as plt

CHUNK = 1024
RATE = 44100
RECORD_SECONDS = 5
FORMAT = pyaudio.paInt16
#if len(sys.argv) < 2:
#    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
#    sys.exit(-1)
#
#wf = wave.open(sys.argv[1], 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()



#plt.plot(t,y)
#plt.show()

# open stream (2)
stream = p.open(format=FORMAT,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=RATE* RECORD_SECONDS)

frames = numpy.array([],numpy.int16)

#for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#    data = stream.read(CHUNK)
#    numpy.append(frames,numpy.frombuffer(data,numpy.int16))
    #frames.append(numpy.frombuffer(data,numpy.int16))


data = stream.read(RATE* RECORD_SECONDS)
frames = numpy.frombuffer(data,numpy.int16)

#numeros = numpy.fromstring(frames,numpy.float32)
#y = stream.read(1000,True)
#y2 = numpy.fromstring(y,numpy.float32)

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()

#waveFile = wave.open('Prueba.wav', 'wb')
#waveFile.setnchannels(1)
#waveFile.setsampwidth(p.get_sample_size(FORMAT))
#waveFile.setframerate(RATE)
#waveFile.writeframes(b''.join(frames))
#waveFile.close()