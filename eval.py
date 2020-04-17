import tensorflow as tf
import librosa
import os
import argparse
from hyperparams import Hyperparameter as hp
from utils import *
from model import RelGAN


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

    source_dir = os.listdir(hp.eval_dir)[x_atr]
    eval_dir = os.path.join(hp.eval_dir, source_dir)
    source_speakers = os.listdir('pickles')

    validation_dir = os.path.join('experiments',
                                  f'converted_{source_speakers[x_atr]}_to_{source_speakers[y_atr]}_alp{alpha}')
    os.makedirs(validation_dir, exist_ok=True)

    print('Loading cached data...')
    speaker_dirs = []
    for f in source_speakers:
        speaker_dirs.append(os.path.join('pickles', f))

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

    print('Generating Validation Data ...')
    labels = np.arange(num_domains)
    for file in glob.glob(eval_dir + '/*.wav'):
        x_labels = np.zeros([1, num_domains])
        y_labels = np.zeros([1, num_domains])
        z_labels = np.zeros([1, num_domains])

        x_labels[0] = np.identity(num_domains)[x_atr]
        y_labels[0] = np.identity(num_domains)[y_atr]
        labels = labels[labels != np.array([x_atr, y_atr])]
        z_atr = np.random.choice(labels, 1)
        z_labels[0] = np.identity(num_domains)[z_atr]
        alpha = np.ones(1) * alpha

        wav, _ = librosa.load(file, sr=hp.rate, mono=True)
        wav *= 1. / max(0.01, np.max(wav))
        wav = wav_padding(wav, sr=hp.rate, frame_period=hp.duration)
        f0, timeaxis, sp, ap = world_decompose(wav, hp.rate, hp.duration)
        f0s_mean_A = np.exp(log_f0s_means[x_atr])
        f0s_mean_B = np.exp(log_f0s_means[y_atr])
        f0s_mean_AB = alpha * f0s_mean_B + (1 - alpha) * f0s_mean_A
        log_f0s_mean_AB = np.log(f0s_mean_AB)
        f0s_std_A = np.exp(log_f0s_stds[x_atr])
        f0s_std_B = np.exp(log_f0s_stds[y_atr])
        f0s_std_AB = alpha * f0s_std_B + (1 - alpha) * f0s_std_A
        log_f0s_std_AB = np.log(f0s_std_AB)
        f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_means[x_atr], std_log_src=log_f0s_stds[y_atr],
                                        mean_log_target=log_f0s_mean_AB, std_log_target=log_f0s_std_AB)

        coded_sp = world_encode_spectral_envelop(sp, hp.rate, hp.num_mceps)
        coded_sp_transposed = coded_sp.T
        coded_sp_norm = (coded_sp_transposed - coded_sps_means[x_atr]) / coded_sps_stds[x_atr]
        coded_sp_norm = np.array([coded_sp_norm])

        inputs = [coded_sp_norm, coded_sp_norm, coded_sp_norm, coded_sp_norm, x_labels, y_labels, z_labels,
                  alpha]
        coded_sp_converted_norm = model(inputs)[0][0].numpy()
        if coded_sp_converted_norm.shape[1] > len(f0):
            coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
        coded_sps_AB_mean = (1 - alpha) * coded_sps_means[x_atr] + alpha * coded_sps_means[y_atr]
        coded_sps_AB_std = (1 - alpha) * coded_sps_stds[x_atr] + alpha * coded_sps_stds[y_atr]
        coded_sp_converted = coded_sp_converted_norm * coded_sps_AB_std + coded_sps_AB_mean
        coded_sp_converted = coded_sp_converted.T
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=hp.rate)
        wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=hp.rate,
                                                 frame_period=hp.duration)
        wav_transformed *= 1. / max(0.01, np.max(np.abs(wav_transformed)))

        librosa.output.write_wav(os.path.join(validation_dir, os.path.basename(file)), wav_transformed, hp.rate)
