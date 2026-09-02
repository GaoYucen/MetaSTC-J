#!/usr/bin/env python3
"""Extract LargeST's first 288 time steps into a compact NumPy cache."""
from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/gla/gla_his_2019.h5")
    parser.add_argument("--output", default="data/gla/gla_his_2019_first288.npy")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    start = time.perf_counter()
    frame = pd.read_hdf(Path(args.source))
    if args.scale <= 0:
        raise ValueError("scale must be positive")
    array = np.ascontiguousarray(frame.head(288).to_numpy(dtype=np.float32).T / args.scale)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, array, allow_pickle=False)
    loaded = np.load(output, allow_pickle=False)
    print({"source_shape": tuple(frame.shape), "cache_shape": tuple(loaded.shape), "scale": args.scale, "original_max_value": float(np.max(array) * args.scale), "scaled_max_value": float(np.max(array)), "original_mean": float(np.mean(array) * args.scale), "scaled_mean": float(np.mean(array)), "max_abs_error": float(np.max(np.abs(loaded-array))), "elapsed_seconds": round(time.perf_counter()-start, 3), "output": str(output)})

if __name__ == "__main__":
    main()
