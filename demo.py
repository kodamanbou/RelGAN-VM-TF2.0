import os
import numpy as np
from hyperparams import Hyperparameter as hp
import argparse
import pyaudio
from demo.encode_worker import encode_worker
from demo.convert_worker import convert_worker
from demo.decode_worker import decode_worker
from multiprocessing import Queue, Process

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
    chunk_in = 1024
    chunk_out = 1024
    stream_in = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=hp.rate,
                       input=True,
                       frames_per_buffer=chunk_in)

    stream_out = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=hp.rate,
                        output=True,
                        frames_per_buffer=chunk_out)

    # Puts results of each workers
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

    while stream_in.is_active():
        data = stream_in.read(chunk_in)
        data = np.frombuffer(data, dtype='int16').astype(np.float64)
        queue_encode.put(data)
        output = queue_output.get()
        output = stream_out.write(output.astype(np.int16).tobytes())

    stream_in.stop_stream()
    stream_in.close()
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()
