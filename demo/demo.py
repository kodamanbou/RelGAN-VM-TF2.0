import tensorflow as tf
import os
from model import RelGAN
from hyperparams import Hyperparameter as hp
from utils import *
import argparse
import pyaudio
import threading
from demo.encode_worker import encode_worker
from demo.convert_worker import convert_worker
from demo.decode_worker import decode_worker
from multiprocessing import Queue, Process


def worker(flag, lock):
    key = str(input('Press q to quit.'))
    while key != 'q':
        key = str(input('Press q to quit.'))

    lock.acquire()
    flag['recording'] = False
    lock.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_label', '-s', dest='source_label', type=int, default=None, required=True,
                        help='source label')
    parser.add_argument('--target_label', '-t', dest='target_label', type=int, default=None, required=True,
                        help='target_label')
    parser.add_argument('--interpolation', '-i', dest='interpolation', type=float, default=1.0,
                        help='interpolation rate')
    args = parser.parse_args()

    x_atr = args.source_label
    y_atr = args.target_label
    alpha = args.interpolation

    p = pyaudio.PyAudio()
    chunk = 2048
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=hp.rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)

    flag = {'recording': True}
    lock = threading.Lock()
    t1 = threading.Thread(target=worker, args=(flag, lock))
    t1.start()

    # Puts results of each workers.
    queue_encode = Queue()
    queue_convert = Queue()
    queue_decode = Queue()
    queue_output = Queue()

    p_encode = Process(target=encode_worker, args=(queue_encode, queue_convert, x_atr, y_atr, alpha))
    p_encode.start()
    p_convert = Process(target=convert_worker,
                        args=(queue_convert, queue_decode, len(os.listdir('pickles')), x_atr, y_atr, alpha))
    p_convert.start()
    p_decode = Process(target=decode_worker, args=(queue_decode, queue_output))
    p_decode.start()

    while flag['recording'] and stream.is_active():
        data = stream.read(chunk)
        queue_encode.put(data)
        output = queue_output.get()
        output = stream.write(output)

    stream.stop_stream()
    stream.close()
    p.terminate()
