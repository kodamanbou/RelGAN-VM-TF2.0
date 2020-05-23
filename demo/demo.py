import tensorflow as tf
import os
from model import RelGAN
from hyperparams import Hyperparameter as hp
from utils import *
import argparse
import pyaudio
import threading
from demo.encode_worker import encode_worker
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

    print('Loading cached data...')
    source_speakers = os.listdir('../pickles')
    speaker_dirs = []
    for f in source_speakers:
        speaker_dirs.append(os.path.join('../pickles', f))

    coded_sps_norms = []
    coded_sps_means = []
    coded_sps_stds = []
    log_f0s_means = []
    log_f0s_stds = []
    for f in speaker_dirs:
        coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
            os.path.join(f, 'cache{}.p'.format(hp.num_mceps)))
        coded_sps_norms.append(coded_sps_A_norm)
        coded_sps_means.append(coded_sps_A_mean)
        coded_sps_stds.append(coded_sps_A_std)
        log_f0s_means.append(log_f0s_mean_A)
        log_f0s_stds.append(log_f0s_std_A)

    num_domains = len(coded_sps_norms)
    model = RelGAN(num_domains)
    latest = tf.train.latest_checkpoint(hp.weights_dir)
    model.load_weights(latest)

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

    queue_encode = Queue()
    queue_convert = Queue()
    queue_decode = Queue()
    queue_output = Queue()

    p_encode = Process(target=encode_worker, args=(queue_encode, queue_convert, x_atr, y_atr, alpha))
    p_encode.start()

    while flag['recording'] and stream.is_active():
        data = stream.read(chunk)
        queue_encode.put(data)
        output = queue_output.get()
        output = stream.write(output)

    stream.stop_stream()
    stream.close()
    p.terminate()
