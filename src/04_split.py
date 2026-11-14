"""
04_split.py - Phase 2: the spatial train / test split (the anti-cheating step).

Plain words:
- Neighbouring pixels look alike and usually share the same label. If we split
  pixels at random, the model could "pass the test" by copying a pixel's
  next-door neighbour from training. That is called leakage.
- So we cut the map into 1 km squares ("blocks"). A whole block goes either to
  TRAIN (the model learns from it) or to TEST (locked away until Phase 4).
- The SAME block assignment is used for every date. A piece of land that is in
  the test set is never seen in training, at any date.
- We mix fairly: blocks are sorted into 3 groups by how damaged they are
  (lightly / medium / heavily), and 30% of each group goes to TEST. Otherwise an
  unlucky shuffle could put almost all intact land on one side.
- Buffer: training pixels within 100 m of a test block are dropped, so no
  training pixel touches (or shares a blurred 20 m band pixel with) test land.
- Training blocks are also dealt into 5 folds for cross-validation in Phase 3.

Outputs:
  data/processed/split/split.tif     per pixel: 0 not cropland, 1 train, 2 test, 3 buffer (unused)
  data/processed/split/cv_fold.tif   per pixel: fold 0-4 for train pixels, 255 elsewhere
  data/processed/split/cv_near.tif   per pixel: which other folds lie within 100 m (one bit per fold)
  results/split_blocks.csv           one row per block
  results/split_summary.json         counts per split, date and class
  results/figures/split_map.png

Run:  .venv/Scripts/python.exe -I src/04_split.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (PROCESSED, RESULTS, SEED, RESOLUTION_M, BLOCK_SIZE_M, TEST_FRACTION,  # noqa: E402
                    BUFFER_M, N_CV_FOLDS, LABEL_LAYERS, CLASSES)
from common import NODATA, block_ids, read_labels  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from scipy import ndimage  # noqa: E402

OUT = PROCESSED / "split"
TRAIN, TEST, BUFFER = 1, 2, 3
N_STRATA = 3


def block_table(labels: dict, bid: np.ndarray) -> pd.DataFrame:
    """One row per block that contains cropland: its size and damaged share at each date."""
    crop = next(iter(labels.values())) != NODATA
    df = pd.DataFrame({"block_id": bid[crop]})
    for d, lab in labels.items():
        df[f"damaged_{d}"] = (lab[crop] == 1)
    g = df.groupby("block_id")
    t = g.size().rename("n_cropland_px").to_frame()
    for d in labels:
        t[f"damaged_share_{d}"] = g[f"damaged_{d}"].mean().round(4)
    t["damaged_share_mean"] = t[[f"damaged_share_{d}" for d in labels]].mean(axis=1)
    return t.reset_index().sort_values("block_id").reset_index(drop=True)


def assign(t: pd.DataFrame) -> pd.DataFrame:
    """Stratified random assignment of whole blocks to train/test, then CV folds for train."""
    rng = np.random.default_rng(SEED)
    # 3 equal-size groups by damage level (rank first so ties cannot break the cut)
    t["stratum"] = pd.qcut(t["damaged_share_mean"].rank(method="first"), N_STRATA, labels=False)
    t["split"], t["cv_fold"] = "train", -1
    for s in range(N_STRATA):
        idx = rng.permutation(t.index[t["stratum"] == s].to_numpy())
        n_test = int(round(TEST_FRACTION * len(idx)))
        t.loc[idx[:n_test], "split"] = "test"
        train_idx = idx[n_test:]
        t.loc[train_idx, "cv_fold"] = np.arange(len(train_idx)) % N_CV_FOLDS   # deal like cards
    return t


def main():
    labels = {d: read_labels(d) for d in LABEL_LAYERS}
    ref = next(iter(labels.values()))
    crop = ref != NODATA
    for lab in labels.values():
        assert ((lab != NODATA) == crop).all(), "cropland mask differs between dates"
    bid = block_ids(ref.shape)

    t = assign(block_table(labels, bid))

    # ---- paint the block decision onto pixels
    is_test_block = np.isin(bid, t.loc[t["split"] == "test", "block_id"])
    is_train_block = np.isin(bid, t.loc[t["split"] == "train", "block_id"])
    # buffer: grow the test blocks by BUFFER_M in every direction; train pixels inside the grown zone are dropped
    r = BUFFER_M // RESOLUTION_M
    near_test = ndimage.binary_dilation(is_test_block, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    split = np.zeros(ref.shape, np.uint8)
    split[crop & is_train_block] = TRAIN
    split[crop & is_train_block & near_test] = BUFFER
    split[crop & is_test_block] = TEST
    fold_of_block = dict(zip(t["block_id"], t["cv_fold"]))
    fold = np.full(ref.shape, 255, np.uint8)
    tr = split == TRAIN
    fold[tr] = np.vectorize(fold_of_block.get)(bid[tr])
    # The same buffer idea between CV folds: cv_near has bit k switched on when a train pixel lies
    # within BUFFER_M of a block of ANOTHER fold k. When fold k is the practice-test fold, those
    # pixels are left out of the practice-training side (see common.cv_splits).
    cv_near = np.zeros(ref.shape, np.uint8)
    for k in range(N_CV_FOLDS):
        fold_k_blocks = np.isin(bid, t.loc[t["cv_fold"] == k, "block_id"])
        near_k = ndimage.binary_dilation(fold_k_blocks, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
        cv_near[tr & near_k & (fold != k)] |= np.uint8(1 << k)

    # ---- safety checks (the script stops if any of these is false)
    train_blocks, test_blocks = set(np.unique(bid[split == TRAIN])), set(np.unique(bid[split == TEST]))
    assert not (train_blocks & test_blocks), "a block is in both train and test"
    assert not (set(np.unique(bid[split == BUFFER])) & test_blocks), "buffer pixels inside a test block"
    assert (split[crop] > 0).all() and (split[~crop] == 0).all(), "every cropland pixel needs exactly one role"
    dist_to_test = ndimage.distance_transform_edt(split != TEST) * RESOLUTION_M   # metres to nearest test pixel
    gap = float(dist_to_test[split == TRAIN].min())
    assert gap > BUFFER_M, f"train pixel only {gap} m from a test pixel"
    for k in range(N_CV_FOLDS):   # folds are also whole blocks
        assert not (set(np.unique(bid[fold == k])) & set(np.unique(bid[(fold != k) & (fold != 255)])))

    # ---- summary numbers
    names = {TRAIN: "train", TEST: "test", BUFFER: "buffer_dropped"}
    summary = {"seed": SEED, "block_size_m": BLOCK_SIZE_M, "buffer_m": BUFFER_M, "test_fraction_target": TEST_FRACTION,
               "n_blocks": {"train": int((t["split"] == "train").sum()), "test": int((t["split"] == "test").sum())},
               "min_train_to_test_distance_m": gap, "pixels": {}, "cv_folds": {}}
    n_crop = int(crop.sum())
    for code, name in names.items():
        m = split == code
        entry = {"n_px": int(m.sum()), "share_of_cropland": round(float(m.sum() / n_crop), 4),
                 "hectares": round(float(m.sum()) * RESOLUTION_M ** 2 / 10_000, 1)}
        for d, lab in labels.items():
            n1 = int((lab[m] == 1).sum())
            entry[d] = {CLASSES[0]: int(m.sum()) - n1, CLASSES[1]: n1, "damaged_share": round(n1 / max(int(m.sum()), 1), 4)}
            if code != BUFFER:
                assert 0 < n1 < int(m.sum()), f"{name} {d} has only one class"
        summary["pixels"][name] = entry
    for k in range(N_CV_FOLDS):
        m = fold == k
        summary["cv_folds"][str(k)] = {"n_blocks": int((t["cv_fold"] == k).sum()), "n_px": int(m.sum()),
                                       **{f"damaged_share_{d}": round(float((lab[m] == 1).mean()), 4) for d, lab in labels.items()}}

    # ---- save
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(PROCESSED / "labels" / f"labels_{next(iter(labels))}.tif") as src:
        prof = src.profile
    summary["cv_px_near_another_fold"] = int((cv_near > 0).sum())
    for name, arr, nod in (("split", split, 0), ("cv_fold", fold, 255), ("cv_near", cv_near, None)):
        prof.update(dtype="uint8", nodata=nod, count=1, compress="deflate")
        with rasterio.open(OUT / f"{name}.tif", "w", **prof) as dst:
            dst.write(arr, 1)
    t.drop(columns="damaged_share_mean").to_csv(RESULTS / "split_blocks.csv", index=False)
    (RESULTS / "split_summary.json").write_text(json.dumps(summary, indent=2))
    figure(split, labels)
    print(json.dumps(summary, indent=2))


def figure(split, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    b = BLOCK_SIZE_M // RESOLUTION_M
    fig, axes = plt.subplots(1, 1 + len(labels), figsize=(6 * (1 + len(labels)), 7.5))
    cols = ["#ffffff", "#4477aa", "#ee7733", "#bbbbbb"]
    axes[0].imshow(split, cmap=ListedColormap(cols), vmin=0, vmax=3, interpolation="nearest")
    axes[0].set_title("Spatial split: whole 1 km blocks")
    axes[0].legend(handles=[Patch(color=cols[1], label="train"), Patch(color=cols[2], label="test (locked)"),
                            Patch(color=cols[3], label="100 m buffer (unused)")], loc="lower right", fontsize=9)
    for ax, (d, lab) in zip(axes[1:], labels.items()):
        show = np.where(lab == NODATA, 0, lab + 1)
        ax.imshow(show, cmap=ListedColormap(["#ffffff", "#228833", "#aa3377"]), vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"UN label {d}")
        ax.legend(handles=[Patch(color="#228833", label="non-affected"), Patch(color="#aa3377", label="damaged")],
                  loc="lower right", fontsize=9)
    for ax in axes:
        for v in range(0, split.shape[1], b):
            ax.axvline(v - 0.5, color="k", lw=0.3, alpha=0.4)
        for h in range(0, split.shape[0], b):
            ax.axhline(h - 0.5, color="k", lw=0.3, alpha=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Khan Younis cropland (UNOSAT-FAO mask). Grid lines = 1 km blocks.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "split_map.png", dpi=130)


if __name__ == "__main__":
    main()
