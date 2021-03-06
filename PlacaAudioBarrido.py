# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:28:06 2018

@author: Axel Lacapmesure
"""

"""PyAudio Example: Play a wave file."""

import pyaudio
import wave
import sys
import numpy as np
from matplotlib import pyplot as plt
import time

# Parámetros de entrada
InNChannels  = 1
InSampleFreq = 192000
InByteDepth  = 3
InBitDepth   = 8*InByteDepth
InDataTypePA = pyaudio.paFloat32
InDataTypeNP = np.float32

ReadTimeLimit    = 10# [s] Límite de la adquisición, en tiempo (raro para valores decimales)
ReadCounterLimit = int(np.ceil(ReadTimeLimit * InSampleFreq)) # Límite de la adquisición, en muestras

ReadPacketSize = 1024 * InNChannels
ReadBufferSize = ReadCounterLimit * InNChannels
ReadBuffer     = np.zeros((ReadBufferSize,), InDataTypeNP)
ReadBufferIdx  = 0
ReadCounter    = 0

# Parámetros de salida
OutNChannels  = 1
OutSampleFreq = InSampleFreq
OutByteDepth  = 3
OutBitDepth   = 8*OutByteDepth
OutDataTypePA = pyaudio.paFloat32
OutDataTypeNP = np.float32

WriteTimeLimit    = ReadTimeLimit # [s] Límite de la escritura, en tiempo
WriteCounterLimit = int(np.ceil(WriteTimeLimit * OutSampleFreq)) # Límite de la escritura, en muestras

WritePacketSize = 1024 * OutNChannels
WritePacket     = np.zeros((WritePacketSize,), OutDataTypeNP)
WriteBufferIdx  = 0
WriteCounter    = 0
WriteBufferSize = OutSampleFreq * OutNChannels # Elijo el largo tal que sea 1 segundo

#%% Genera la forma de onda a escribir

#Freq   = 500 # [Hz]
#Amp    = 1
#
#WriteBufferSize = OutSampleFreq * OutNChannels # Elijo el largo tal que sea 1 segundo
#WriteBuffer     = np.array(Amp * np.sin(2*np.pi*Freq * np.arange(WriteBufferSize)/OutSampleFreq)).astype(OutDataTypeNP)

#%% Define callbacks

def read(in_data, frame_count, time_info, status):
    global InNChannels
    
    global ReadBuffer
    global ReadBufferIdx
    global ReadPacketSize
    global ReadCounter
    global ReadLimit
    
    global ReadStartTime
    global ReadEndTime
    
    if ReadCounter == 0:
        ReadStartTime = time_info
    
    # Verifica si el paquete a escribir "rodea" al buffer.
    if ReadBufferIdx + ReadPacketSize > ReadBufferSize:
        # Caso afirmativo, primero toma los datos desde ReadBufferIdx hasta el
        # final del buffer y luego los datos remanentes.
        FirstPacketSize = ReadBufferSize - ReadBufferIdx
        SecondPacketSize = ReadPacketSize - FirstPacketSize
        # Copia los dos paquetes
        ReadBuffer[ReadBufferIdx:ReadBufferIdx+FirstPacketSize] = np.frombuffer(in_data, InDataTypeNP, FirstPacketSize, 0)
        ReadBuffer[0:SecondPacketSize] = np.frombuffer(in_data, InDataTypeNP, SecondPacketSize, FirstPacketSize)
    else:
        ReadBuffer[ReadBufferIdx:ReadBufferIdx+ReadPacketSize] = np.frombuffer(in_data, InDataTypeNP)
    
    ReadBufferIdx = (ReadBufferIdx + ReadPacketSize) % ReadBufferSize
    ReadCounter += ReadPacketSize / InNChannels
    
    
    if ReadCounter >= ReadCounterLimit:
        ReadEndTime = time_info
        return (in_data, pyaudio.paComplete)
    else:
        return (in_data, pyaudio.paContinue)


def write(in_data, frame_count, time_info, status):
    global WriteBuffer
    global WriteBufferIdx
    global WritePacket
    global WritePacketSize
    global WriteCounter
    global WriteCounterLimit
    
    global WriteStartTime
    global WriteEndTime
    
    if WriteCounter == 0:
        WriteStartTime = time_info
    
    # Verifica si el paquete a escribir "rodea" al buffer.
    if WriteBufferIdx + WritePacketSize > WriteBufferSize:
        # Caso afirmativo, primero toma los datos desde WriteBufferIdx hasta el
        # final del buffer y luego los datos remanentes.
        FirstPacketSize = WriteBufferSize - WriteBufferIdx
        SecondPacketSize = WritePacketSize - FirstPacketSize
        # Copia los dos paquetes
        WritePacket[0:FirstPacketSize] = WriteBuffer[WriteBufferIdx:WriteBufferIdx + FirstPacketSize]
        WritePacket[FirstPacketSize:WritePacketSize] = WriteBuffer[0:SecondPacketSize]
    else:
        WritePacket = WriteBuffer[WriteBufferIdx:WriteBufferIdx+WritePacketSize]
    
    WriteBufferIdx = (WriteBufferIdx + WritePacketSize) % WriteBufferSize
    WriteCounter   += WritePacketSize
    

    if WriteCounter >= WriteCounterLimit:
        WriteEndTime = time_info
        return (WritePacket.tobytes(), pyaudio.paComplete)
    else:
        return (WritePacket.tobytes(), pyaudio.paContinue)

#%% Pruebo la función callback

#InTest = np.arange(ReadBufferSize*1.5).astype(InDataTypeNP)
#frame_count = ReadPacketSize
#time_info = {'current_time': time.time()}
#status = None
#
#for i in range(len(InTest)//ReadPacketSize):
#    read(InTest[i*ReadPacketSize:(i+1)*ReadPacketSize].tobytes(), frame_count, time_info, status)
#
#plt.plot(ReadBuffer)

#%% Ejecuta el stream

p = pyaudio.PyAudio()

OutStream = p.open(format = OutDataTypePA,
                   channels = InNChannels,
                   rate = OutSampleFreq,
                   frames_per_buffer = WritePacketSize,
                   output=True,
                   stream_callback = write)

InStream = p.open(format = InDataTypePA,
                  channels = OutNChannels,
                  rate = InSampleFreq,
                  frames_per_buffer = ReadPacketSize,
                  input=True,
                  stream_callback = read)

freq_start = 400
freq_stop = 500
freq_N = 1
Amp    = 1
frequencies = np.logspace(np.log10(freq_start),np.log10(freq_stop),freq_N)


Respuesta = np.zeros((freq_N,))
ind=0
for Freq in frequencies:
    ReadBufferIdx  = 0
    ReadBuffer     = np.zeros((ReadBufferSize,), InDataTypeNP)
    ReadCounter = 0
    WriteBufferIdx  = 0
    WriteCounter = 0
    WriteBuffer     = np.array(Amp * np.sin(2*np.pi*Freq * np.arange(WriteBufferSize)/OutSampleFreq)).astype(OutDataTypeNP)
    


    OutStream.start_stream()
    InStream.start_stream()
    
    
    while OutStream.is_active() | InStream.is_active():
        time.sleep(0.1)
    
    OutStream.stop_stream()
    InStream.stop_stream()
    time.sleep(0.1)
    
    Respuesta[ind]= np.std(ReadBuffer[ReadBufferSize//2:ReadBufferSize])
    ind = ind+1
    
OutStream.close()
InStream.close()
   
    
# close PyAudio (5)
p.terminate()

#%% Grafico salida

plt.semilogx(frequencies,Respuesta,'.')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Vrms')
plt.grid()
plt.title('Respuesta en frecuencia de la placa de sonido (muestreo {} kHz)'.format(InSampleFreq/1000))