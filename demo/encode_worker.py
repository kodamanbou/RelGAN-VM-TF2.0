import os
from hyperparams import Hyperparameter as hp
from utils import *
from multiprocessing import Queue
from demo.acoustic_feature import AcousticFeature


def encode_worker(queue_input: Queue, queue_output: Queue, x_atr, y_atr, alpha):
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

    while True:
        wav = queue_input.get()
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
        coded_sp_norm = np.array([coded_sp_norm], dtype=np.float32)

        coded_sps_AB_mean = (1 - alpha) * coded_sps_means[x_atr] + alpha * coded_sps_means[y_atr]
        coded_sps_AB_std = (1 - alpha) * coded_sps_stds[x_atr] + alpha * coded_sps_stds[y_atr]

        queue_output.put(AcousticFeature(f0_converted, coded_sp_norm, ap, coded_sps_AB_mean, coded_sps_AB_std))
