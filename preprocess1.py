import os
import time
from multiprocessing import Pool
from utils import *
from hyperparams import Hyperparameter as hp


def process1(folder):
    """
    Trim the silence.
    :param folder: folder of each speakers.
    :return:
    """
    divs = 64
    X = []
    for file in glob.glob(folder + '/*.wav'):
        wav, _ = librosa.load(file, sr=hp.rate, mono=True)
        wav *= 1. / max(0.01, np.max(np.abs(wav)))

        wav_splitted = librosa.effects.split(wav, top_db=48)

        export_dir = folder.split(os.sep)[1]
        os.makedirs(os.path.join(hp.data_processed, export_dir), exist_ok=True)

        for s in range(wav_splitted.shape[0]):
            x = wav[wav_splitted[s][0]:wav_splitted[s][1]]
            X = np.concatenate([X, x], axis=0)
    X *= 1. / max(0.01, np.max(np.abs(X)))
    wavlen = X.shape[0]
    crop_size = wavlen // divs
    start = 0
    for i in range(divs):
        sub = 0
        if i == divs - 1:
            sub = X[start:]
        else:
            sub = X[start:start + crop_size]

        start += crop_size
        sub = sub.astype(np.float32)
        librosa.output.write_wav(
            os.path.join(hp.data_processed, export_dir, "{}_".format(i) + os.path.basename(folder) + ".wav"), sub,
            hp.rate)


if __name__ == '__main__':
    folders = glob.glob(hp.data_src + '/*')
    print('Preprocess 1...')
    TIME = time.time()
    cores = min(len(folders), 4)
    p = Pool(cores)
    p.map(process1, folders)
    p.close()
    print(f'Preprocess 1 done in {time.time() - TIME} t.')
