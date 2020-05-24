from multiprocessing import Queue
from hyperparams import Hyperparameter as hp
from utils import *


def decode_worker(queue_input: Queue, queue_output: Queue):
    while True:
        feature = queue_input.get()
        coded_sp_converted = feature.coded_sp_norm
        decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=hp.rate)
        wav_transformed = world_speech_synthesis(f0=feature.f0, decoded_sp=decoded_sp_converted, ap=feature.ap,
                                                 fs=hp.rate,
                                                 frame_period=hp.duration)
        wav_transformed *= 1. / max(0.01, np.max(np.abs(wav_transformed)))

        queue_output.put(wav_transformed)
