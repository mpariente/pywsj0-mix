from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm
import argparse
FS_ORIG = 16000

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--wsj0_path", default="../")
parser.add_argument("-o", "--output_folder", default="simulated_data")
parser.add_argument("-n", "--n_src", default=4, type=int)
parser.add_argument("-sr", "--samplerate", default=8000, type=int)
parser.add_argument("--len_mode", nargs="+", type=list, default=["min", "max"])
args = parser.parse_args()

# Read activlev file. Build {utt_id: activlev} dict
activlev_df = pd.concat([
    pd.read_csv(f"metadata/activlev/activlev_{cond}.txt", delimiter=" ", names=["utt", "alev"], index_col=False)
    for cond in ["tr", "cv", "tt"]
])
activlev_dic = dict(zip(list(activlev_df.utt), list(activlev_df.alev)))


for cond in ["tr", "cv", "tt"]:
    # Output folders (wav8k-16k/min-max/tr-cv-tt/mix-src{i})
    base = Path(args.output_folder) / f"{args.n_src}speakers" / f"wav{args.samplerate // 1000}"
    min_mix_folder =  base / "min"/  cond / "mix"
    min_src_folders = [base / "min"/ cond /  f"s{i+1}" for i in range(args.n_src)]
    max_mix_folder =  base / "max"/  cond / "mix"
    max_src_folders = [base / "max"/ cond /  f"s{i+1}" for i in range(args.n_src)]
    for p in min_src_folders + max_src_folders + [min_mix_folder, max_mix_folder]:
        p.mkdir(parents=True, exist_ok=True)

    # Read SNR scales file
    header = [x for t in zip([f"s_{i}" for i in range(args.n_src)], [f"snr_{i}" for i in range(args.n_src)]) for x in t]
    mix_df = pd.read_csv(f"metadata/mix_{args.n_src}_spk_{cond}.txt", delimiter=" ", names=header, index_col=False)

    for idx in tqdm(range(len(mix_df))):
        sources = [sf.read(Path(args.wsj0_path) / mix_df[f"s_{i}"][idx], dtype="float32")[0] for i in range(args.n_src)]
        snrs = [mix_df[f"snr_{i}"][idx] for i in range(args.n_src)]

        resampled_sources = [resample_poly(s, args.samplerate, FS_ORIG) for s in sources]
        min_len, max_len = min([len(s) for s in resampled_sources]), max([len(s) for s in resampled_sources])
        padded_sources = [np.hstack((s, np.zeros(max_len - len(s)))) for s in resampled_sources]

        activlev_scales = [activlev_dic[mix_df[f"s_{i}"][idx].split("/")[-1].replace(".wav", "")] for i in range(args.n_src)]
        # activlev_scales = [np.sqrt(np.mean(s**2)) for s in resampled_sources]  # If no activlev file
        scaled_sources = [s / scale * 10 ** (x/20) for s, scale, x in zip(padded_sources, activlev_scales, snrs)]

        sources_np = np.stack(scaled_sources, axis=0)
        mix_np = np.sum(sources_np, axis=0)

        gain = np.max([1., np.max(np.abs(mix_np)), np.max(np.abs(sources_np))]) / 0.9
        mix_np /= gain
        sources_np /= gain

        # Merge filenames for mixture name.
        pp = lambda x: x.split('/')[-1].replace(".wav", "") if isinstance(x, str) else str(round(x, 4))
        filename = "_".join([pp(mix_df[u][idx]) for u in header]) + ".wav"

        if "max" in args.len_mode:
            sf.write(max_mix_folder / filename, mix_np, samplerate=args.samplerate)
            for s_fold, src_np in zip(max_src_folders, sources_np):
                sf.write(s_fold / filename, src_np, samplerate=args.samplerate)
        if "min" in args.len_mode:
            sf.write(min_mix_folder / filename, mix_np[:min_len], samplerate=args.samplerate)
            for s_fold, src_np in zip(min_src_folders, sources_np):
                sf.write(s_fold / filename, src_np[:min_len], samplerate=args.samplerate)


