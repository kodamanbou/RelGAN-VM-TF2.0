import tensorflow as tf
from multiprocessing import Queue
from model import RelGAN
from utils import *
from hyperparams import Hyperparameter as hp


def convert_worker(queue_input: Queue, queue_output: Queue, num_domains, x_atr, y_atr, alpha):
    labels = np.arange(num_domains)
    x_labels = np.zeros([1, num_domains], dtype=np.float32)
    y_labels = np.zeros([1, num_domains], dtype=np.float32)
    z_labels = np.zeros([1, num_domains], dtype=np.float32)

    x_labels[0] = np.identity(num_domains)[x_atr]
    y_labels[0] = np.identity(num_domains)[y_atr]
    labels = labels[labels != x_atr]
    labels = labels[labels != y_atr]
    z_atr = np.random.choice(labels, 1)
    z_labels[0] = np.identity(num_domains)[z_atr]
    alpha = np.ones(1) * alpha

    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    model = RelGAN(num_domains)
    latest = tf.train.latest_checkpoint(hp.weights_dir)
    model.load_weights(latest)

    while True:
        features = queue_input.get()
        coded_sp_norm = features.coded_sp_norm
        inputs = [coded_sp_norm, x_labels, y_labels, alpha]
        coded_sp_converted_norm = model.generate(inputs)[0].numpy()
        if coded_sp_converted_norm.shape[1] > len(features.f0):
            coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]

        coded_sp_converted = coded_sp_converted_norm * features.coded_sps_std + features.coded_sps_mean
        coded_sp_converted = coded_sp_converted.T
        features.coded_sp_norm = np.ascontiguousarray(coded_sp_converted)

        queue_output.put(features)
