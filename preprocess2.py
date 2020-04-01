import os
import time
from multiprocessing import Pool
from utils import *
from hyperparams import Hyperparameter as hp


def process2(folder):
    train_A_dir = os.path.join(hp.data_processed, folder)
    exp_dir = os.path.join('pickles', folder)
    os.makedirs(exp_dir, exist_ok=True)

    print('Loading Wavs...')

    start_time = time.time()

    wavs_A = load_wavs(wav_dir=train_A_dir, sr=hp.rate)

    print('Extracting acoustic features...')

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs=wavs_A, fs=hp.rate,
                                                                     frame_period=hp.duration, coded_dim=hp.num_mceps)

    print('Calculating F0 statistics...')

    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)

    print('Log Pitch A')
    print('Mean: %f, Std: %f' % (log_f0s_mean_A, log_f0s_std_A))

    print('Normalizing data...')

    coded_sps_A_transposed = transpose_in_list(lst=coded_sps_A)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_A_transposed)

    print('Saving data...')
    save_pickle(os.path.join(exp_dir, 'cache{}.p'.format(hp.num_mceps)),
                (coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A))

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
        time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))


if __name__ == '__main__':
    print('Preprocess 2...')
    folders = os.listdir(hp.data_processed)
    cores = min(len(folders), 4)
    TIME = time.time()
    p = Pool(cores)
    p.map(process2, folders)
    p.close()
    print(f'Preprocess 2 done in {time.time() - TIME} t.')
