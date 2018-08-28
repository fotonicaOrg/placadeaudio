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

#if len(sys.argv) < 2:
#    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
#    sys.exit(-1)
#
#wf = wave.open(sys.argv[1], 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

fwav = 44000
t = numpy.arange(0,fwav)
fA = 440
bitrate = 2**32
y = (0.5*numpy.sin(2*numpy.pi*fA*t/fwav)).astype(numpy.float32)




#plt.plot(t,y)
#plt.show()

# open stream (2)
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fwav,
                output=True)

# read data
#data = wf.readframes(CHUNK)

# play stream (3)



#for i in numpy.arange(0,len(y)//CHUNK):
#    data = y[(CHUNK*i):(CHUNK*(i+1))]
#    stream.write(data)
    
stream.write(y)

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()