import tensorflow as tf
import numpy as np
import glob
import os
from utils import load_pickle, sample_train_data
from hyperparams import Hyperparameter as hp
from model import RelGAN


@tf.function
def train_step(inputs):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        outputs = model(inputs)
        generation_B = outputs[0]
        generation_B2 = outputs[1]


if __name__ == '__main__':
    print('Loading cached data...')
    coded_sps_norms = []
    coded_sps_means = []
    coded_sps_stds = []
    log_f0s_means = []
    log_f0s_stds = []
    for f in glob.glob(hp.data_processed + '/*'):
        coded_sps_norm, coded_sps_mean, coded_sps_std, log_f0s_mean, log_f0s_std = load_pickle(
            os.path.join(f, f'pickles/cache{hp.num_mceps}.p'))
        coded_sps_norms.append(coded_sps_norm)
        coded_sps_means.append(coded_sps_mean)
        coded_sps_stds.append(coded_sps_std)
        log_f0s_means.append(log_f0s_mean)
        log_f0s_stds.append(log_f0s_std)

    num_domains = len(coded_sps_norms)
    model = RelGAN(num_domains)

    iteration = 0
    while iteration < hp.num_iterations:
        if iteration % 10000 == 0:
            hp.lambda_triangle *= 0.9
            hp.lambda_backward *= 0.9

        hp.generator_lr *= 0.99999
        hp.discriminator_lr *= 0.99999
        x, x2, x_atr, y, y_atr, z, z_atr = sample_train_data(dataset_A=coded_sps_norms, nBatch=hp.batch_size)

        x_labels = np.zeros([hp.batch_size, num_domains])
        y_labels = np.zeros([hp.batch_size, num_domains])
        z_labels = np.zeros([hp.batch_size, num_domains])
        for b in range(hp.batch_size):
            x_labels[b] = np.identity(num_domains)[x_atr[b]]
            y_labels[b] = np.identity(num_domains)[y_atr[b]]
            z_labels[b] = np.identity(num_domains)[z_atr[b]]

        rnd = np.random.randint(2)
        alpha = np.random.uniform(0, 0.5, size=hp.batch_size) if rnd == 0 else np.random.uniform(0.5, 1.0,
                                                                                                 size=hp.batch_size)

        inputs = [x, x2, y, z, x_labels, y_labels, z_labels, alpha]
        # training.
        train_step(inputs)
