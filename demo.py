import tensorflow as tf
import os
from model import RelGAN
from hyperparams import Hyperparameter as hp
from utils import *
import pyaudio
import threading


def worker(flag, lock):
    key = str(input('Press q to quit.'))
    while key != 'q':
        key = str(input('Press q to quit.'))

    lock.acquire()
    flag['recording'] = False
    lock.release()


if __name__ == '__main__':
    p = pyaudio.PyAudio()
    chunk = 2048
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)

    flag = {'recording': True}
    lock = threading.Lock()
    t1 = threading.Thread(target=worker, args=(flag, lock))
    t1.start()

    while flag['recording'] and stream.is_active():
        data = stream.read(chunk)
        output = stream.write(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
