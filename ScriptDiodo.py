# -*- coding: utf-8 -*-

import sys
import importlib

if "PlacaAudio" not in sys.modules:
    import PlacaAudio as pa
else:
    importlib.reload(pa)

import scipy.signal
import numpy as np
from matplotlib import pyplot as plt

print('Ejecutando')

sample_freq = 192000

signal_amp  = 2
signal_freq = 1000
signal_len  = sample_freq * 2
total_time = 1

channel_left = pa.function_generator("sine", signal_freq, sample_freq, signal_amp,signal_len,np.float32)
channel_right = channel_left

(out_left, out_right) = pa.play_rec(
                            in_data = (channel_left, channel_right),
                            sample_freq = sample_freq,
                            read_time_limit = total_time,
                            write_time_limit = total_time,
                            )

# me quedo con el centro del vector
Imin = len(out_left)//2
Imax = 9*len(out_left)//10
L = out_left[Imin:Imax]
R = out_right[Imin:Imax]

calibracion = 1/.414    
resistencia = 1000
I = L*calibracion/resistencia * 1000 #corriente en mA
V = (R-L)*calibracion

plt.figure()
plt.plot(L*calibracion, label = "L")
plt.plot(R*calibracion, label = "R")
plt.legend()

plt.figure()
plt.plot(V,I)

sort_idx = np.argsort(V)
V = V[sort_idx]
I = I[sort_idx]

I_smooth = scipy.signal.savgol_filter(I, 2001, 2)

Vmean = np.mean(L[np.abs(R)<0.025])*calibracion

plt.figure()
#plt.plot(V,I)
#plt.figure()
plt.plot(V + Vmean, I_smooth-Vmean/resistencia*1000)
plt.grid()
