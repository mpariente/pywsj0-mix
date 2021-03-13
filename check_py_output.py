import soundfile as sf
from tqdm import tqdm
import os
import glob
import argparse
import numpy as np
from numpy.testing import assert_allclose

FS_ORIG = 16000
MIN_PREC = 3.1e-5

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--matlab_path", default="/mnt/beegfs/pul51/zaf67/DATA/matlab_wsj0_mix/2speakers/wav8k/min/tt")
parser.add_argument("-o", "--python_path", default="/mnt/beegfs/pul51/zaf67/DATA/wsj0-mix/again/2speakers/wav8k/min/tt")
args = parser.parse_args()

matlab_glob = glob.glob(os.path.join(args.matlab_path, "**/*.wav"), recursive=True)
matlab_wav = list(map(lambda x: x.replace(args.matlab_path, "")[1:], matlab_glob))

def snr_fn(x, y):
    noise = y - x
    return 10 * np.log10(np.sum(x**2)) - 10 * np.log10(np.sum(noise**2))

for mwp in tqdm(matlab_wav):
    x1, fs1 = sf.read(os.path.join(args.matlab_path, mwp), dtype="float64")
    try:
        x2, fs2 = sf.read(os.path.join(args.python_path, mwp), dtype="float64")
    except:
        print("No : ", mwp)
        continue
    # try:
    #     assert_allclose(x1, x2, atol=5 * MIN_PREC)
    # except:
    # print("Didn't pass", snr_fn(x1, x2))
    snr = snr_fn(x1, x2)
    if snr < 80:
        print("SNR (dB) : ", snr )




