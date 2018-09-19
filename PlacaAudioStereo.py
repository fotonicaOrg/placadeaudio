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

#%% Funciones genéricas

def multiplex(channel_left, channel_right):
    if len(channel_left) != len(channel_right):
        raise(RuntimeError,'Los dos canales deben tener el mismo largo')
    
    out_length = len(channel_left)*2
    
    out = np.zeros((out_length,),channel_left.dtype)
    out[0:out_length:2] = channel_left
    out[1:out_length:2] = channel_right

    return out

def demultiplex(read):
    read_length = len(read)
    channel_left = read[0:read_length:2]
    channel_right = read[1:read_length:2]
    
    return (channel_left, channel_right)

#%% Configuración

def configure_input(
        n_channels,
        sample_freq,
        time_limit,
        buffer_size = None,
        packet_size_in_frames = 1024,
        data_type_NP = np.float32,
        data_type_PA = pyaudio.paFloat32,
        byte_depth = 3
        ):
    
    global InNChannels, InSampleFreq, InByteDepth, InBitDepth, InDataTypePA, InDataTypeNP
    global ReadTimeLimit, ReadCounterLimit, ReadPacketSize, ReadPacketSizeInFrames, ReadPacket, ReadBufferSize, ReadBuffer
    global ReadBufferIdx, ReadCounter
    
    # Parámetros de entrada
    InNChannels  = n_channels
    InSampleFreq = sample_freq
    InByteDepth  = byte_depth
    InBitDepth   = 8 * byte_depth
    InDataTypePA = data_type_PA
    InDataTypeNP = data_type_NP
    
    ReadTimeLimit    = time_limit # [s] Límite de la adquisición, en tiempo (raro para valores decimales)
    ReadCounterLimit = int(np.ceil(time_limit * sample_freq * n_channels)) # Límite de la adquisición, en muestras
    
    ReadPacketSizeInFrames = packet_size_in_frames
    ReadPacketSize = packet_size_in_frames * n_channels
    ReadPacket     = np.zeros((ReadPacketSize,), data_type_NP)
    
    if buffer_size == None:
        ReadBufferSize = ReadCounterLimit + ReadPacketSize
    else:
        ReadBufferSize = buffer_size
    ReadBuffer     = np.zeros((ReadBufferSize,), data_type_NP)
    ReadBufferIdx  = 0
    ReadCounter    = 0
    
    def read(
            in_data,
            frame_count,
            time_info,
            status
            ):
        
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
        ReadCounter += ReadPacketSize
        
        
        if ReadCounter >= ReadCounterLimit:
            ReadEndTime = time_info
            return (in_data, pyaudio.paComplete)
        else:
            return (in_data, pyaudio.paContinue)
    
    return read

def configure_output(
        n_channels,
        sample_freq,
        time_limit,
        buffer_size,
        packet_size_in_frames = 1024,
        data_type_NP = np.float32,
        data_type_PA = pyaudio.paFloat32,
        byte_depth = 3
        ):
    
    global OutNChannels, OutSampleFreq, OutByteDepth, OutBitDepth, OutDataTypePA, OutDataTypeNP
    global WriteTimeLimit, WriteCounterLimit, WritePacketSize, WritePacketSizeInFrames, WritePacket, WriteBufferSize, WriteBuffer
    global WriteBufferIdx, WriteCounter
    
    # Parámetros de entrada
    OutNChannels  = n_channels
    OutSampleFreq = sample_freq
    OutByteDepth  = byte_depth
    OutBitDepth   = 8 * byte_depth
    OutDataTypePA = data_type_PA
    OutDataTypeNP = data_type_NP
    
    WriteTimeLimit    = time_limit # [s] Límite de la adquisición, en tiempo (raro para valores decimales)
    WriteCounterLimit = int(np.ceil(time_limit * sample_freq * n_channels)) # Límite de la adquisición, en muestras
    
    WritePacketSizeInFrames = packet_size_in_frames
    WritePacketSize = packet_size_in_frames * n_channels
    WritePacket     = np.zeros((WritePacketSize,), data_type_NP)
    
    if buffer_size == None:
        WriteBufferSize = WriteCounterLimit + WritePacketSize
    else:
        WriteBufferSize = buffer_size
    WriteBuffer     = np.zeros((WriteBufferSize,), data_type_NP)
    WriteBufferIdx  = 0
    WriteCounter    = 0
    
    def write(
            in_data,
            frame_count,
            time_info,
            status
            ):
        
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
            WritePacket[0:WritePacketSize] = WriteBuffer[WriteBufferIdx:WriteBufferIdx+WritePacketSize]
        
        WriteBufferIdx = (WriteBufferIdx + WritePacketSize) % WriteBufferSize
        WriteCounter   += WritePacketSize
        
    
        if WriteCounter >= WriteCounterLimit:
            WriteEndTime = time_info
            return (WritePacket.tobytes(), pyaudio.paComplete)
        else:
            return (WritePacket.tobytes(), pyaudio.paContinue)
    
    return write

def reset_read_buffer():
    global ReadBuffer
    global ReadBufferIdx
    global ReadCounter
    
    ReadBuffer     = np.zeros((ReadBufferSize,), InDataTypeNP)
    ReadBufferIdx  = 0
    ReadCounter    = 0

def reset_write_buffer():
    global WriteBuffer
    global WriteBufferIdx
    global WriteCounter
    
    WriteBuffer     = np.zeros((WriteBufferSize,), OutDataTypeNP)
    WriteBufferIdx  = 0
    WriteCounter    = 0

#%% Pruebo la función callback

def test_callbacks(
        read_callback,
        write_callback
    ):
    
    WriteResult = []
    ReadStatus = pyaudio.paContinue
    WriteStatus = pyaudio.paContinue
    
    while ReadStatus is pyaudio.paContinue | WriteStatus is pyaudio.paContinue:
        time_info = {'current_time': time.time()}
        
        (WriteArray, WriteStatus) = write_callback(None, WritePacketSizeInFrames, time_info, None)
        (_, ReadStatus) = read_callback(WriteArray, ReadPacketSizeInFrames, time_info, None)
        
        WriteResult = np.append(WriteResult, np.frombuffer(WriteArray, OutDataTypeNP))
    
    print("Fin de prueba. ReadCounter = {}, WriteCounter = {}".format(ReadCounter,WriteCounter))

#%% Ejecuta el stream

def play_rec(
        in_data,
        sample_freq,
        read_time_limit,
        write_time_limit,
        wait_between_streams = 0,
        test = False
        ):
    
    global WriteBuffer
    
    # Detecto los datos de canal y otra info a partir del array de entrada
    if type(in_data) == tuple:
        n_channels = len(in_data)
    else:
        n_channels = 1
    
    # Demultiplexo canales
    if n_channels == 1:
        in_length = len(in_data)
    elif n_channels == 2:
        channel_left  = in_data[0]
        channel_right = in_data[1]
        
        if len(channel_left) != len(channel_right):
            raise RuntimeError('Los canales deben tener el mismo largo.')
        else:
            in_length = len(channel_left)
    else:
        raise RuntimeError('La cantidad de canales debe ser 1 o 2.')
    
    write_buffer_size = in_length * n_channels
    
    # Configuro
    write = configure_output(n_channels = n_channels,
                             sample_freq = sample_freq,
                             time_limit = write_time_limit,
                             buffer_size = write_buffer_size,
                             packet_size_in_frames = 1024,
                             data_type_NP = np.float32,
                             data_type_PA = pyaudio.paFloat32,
                             byte_depth = 3)
    
    read = configure_input(n_channels = n_channels,
                           sample_freq = sample_freq,
                           time_limit = read_time_limit,
                           buffer_size = None,
                           packet_size_in_frames = 1024,
                           data_type_NP = np.float32,
                           data_type_PA = pyaudio.paFloat32,
                           byte_depth = 3)        
    
    # Multiplexo la salida
    if n_channels == 1:
        WriteBuffer[0:write_buffer_size] = in_data
    elif n_channels == 2:
        WriteBuffer[0:write_buffer_size] = multiplex(channel_left, channel_right)
    
    # Ejecución en modo de prueba
    if test:
        test_callbacks(read, write)
        
    # Ejecución en modo normal
    else:
        # Inicializo pyaudio y streams
        p = pyaudio.PyAudio()
        
        OutStream = p.open(format = OutDataTypePA,
                           channels = OutNChannels,
                           rate = OutSampleFreq,
                           frames_per_buffer = WritePacketSizeInFrames,
                           output = True,
                           stream_callback = write)
        
        InStream = p.open(format = InDataTypePA,
                          channels = InNChannels,
                          rate = InSampleFreq,
                          frames_per_buffer = ReadPacketSizeInFrames,
                          input = True,
                          stream_callback = read)
        
        # Inicio los streams
        OutStream.start_stream()
        if wait_between_streams != 0:
            time.sleep(wait_between_streams)
        InStream.start_stream()
        
        # Espero a que termine
        while OutStream.is_active() | InStream.is_active():
            time.sleep(0.1)
        
        # Termino los streams
        OutStream.stop_stream()
        InStream.stop_stream()
        
        time.sleep(0.1)
        
        # Cierro streams y termino pyaudio
        OutStream.close()
        InStream.close()
        p.terminate()
    
    # Demodulo los dos canales
    if n_channels == 1:
        return ReadBuffer
    elif n_channels == 2:
        return demultiplex(ReadBuffer)


if __name__ == "__main__":
    
    print('Ejecutando')
    
    sample_freq = 44100
    
    signal_amp  = 1
    signal_freq = 400
    signal_len  = sample_freq * 2
    
    channel_right  = np.array(signal_amp * np.sin(2*np.pi*signal_freq * np.arange(signal_len)/sample_freq)).astype(np.float32)
    channel_left = np.zeros((signal_len,), np.float32)

    (out_left, out_right) = play_rec(in_data = (channel_left, channel_right),
                                     sample_freq = sample_freq,
                                     read_time_limit = 1,
                                     write_time_limit = 1,
                                     test = True)
    
    plt.plot(out_left)
    plt.plot(out_right)
