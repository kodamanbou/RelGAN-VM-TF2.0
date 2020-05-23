import dataclasses
import numpy as np


@dataclasses.dataclass
class AcousticFeature:
    f0: np.ndarray
    coded_sp_norm: np.ndarray
    ap: np.ndarray
    coded_sps_mean: np.ndarray
    coded_sps_std: np.ndarray
