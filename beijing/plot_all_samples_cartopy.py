import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = CURRENT_DIR / "data" / "all_samples_20260051_2026-06-21-12-00-00.npy"
DEFAULT_GT = CURRENT_DIR / "data" / "real_track_20260051_2026062120.txt"
DEFAULT_OUTPUT = CURRENT_DIR / "plot" / "all_samples_20260051_2026062112_cartopy_with_gt.png"
LEAD_HOURS = [12, 24, 36, 48, 60, 72, 96, 120]


def valid_track(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    valid = np.isfinite(points).all(axis=1)
    valid &= ~((np.abs(points[:, 0]) < 1e-8) & (np.abs(points[:, 1]) < 1e-8))
    return points[valid]


def map_extent(sample: np.ndarray, gt_track: np.ndarray | None = None) -> list[float]:
    points = sample.reshape(-1, 2)
    if gt_track is not None and len(gt_track) > 0:
        points = np.vstack([points, gt_track])
    points = valid_track(points)
    if len(points) == 0:
        return [100, 160, 0, 50]

    lats = points[:, 0]
    lons = points[:, 1]
    lat_pad = max(2.0, (lats.max() - lats.min()) * 0.12)
    lon_pad = max(2.0, (lons.max() - lons.min()) * 0.12)
    return [
        max(95.0, lons.min() - lon_pad),
        min(180.0, lons.max() + lon_pad),
        max(0.0, lats.min() - lat_pad),
        min(60.0, lats.max() + lat_pad),
    ]


def add_lead_labels(ax, track: np.ndarray, color: str) -> None:
    for idx, (lat, lon) in enumerate(track):
        if idx >= len(LEAD_HOURS):
            break
        ax.annotate(
            f"{LEAD_HOURS[idx]}h",
            (lon, lat),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            color=color,
            transform=ccrs.PlateCarree(),
        )


def parse_gt_track(gt_path: Path, base_time_bjt: str) -> np.ndarray:
    base_dt = datetime.strptime(base_time_bjt, "%Y-%m-%d %H:%M:%S")
    wanted = {base_dt + timedelta(hours=hour): hour for hour in LEAD_HOURS}
    found: dict[int, tuple[float, float]] = {}

    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            point_dt = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S")
            lead_hour = wanted.get(point_dt)
            if lead_hour is None:
                continue
            found[lead_hour] = (float(parts[2]), float(parts[3]))

    missing = [hour for hour in LEAD_HOURS if hour not in found]
    if missing:
        raise ValueError(f"GT file missing lead-hour points: {missing}")

    return np.array([found[hour] for hour in LEAD_HOURS], dtype=float)


def plot_sample(
    all_samples: np.ndarray,
    sample_index: int,
    output: Path,
    gt_track: np.ndarray | None,
) -> None:
    sample = all_samples[sample_index]
    if sample.ndim != 3 or sample.shape[-1] != 2:
        raise ValueError(f"Expected sample shape (lead, channel, 2), got {sample.shape}")

    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 9), dpi=180)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent(map_extent(sample, gt_track), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="#dff3fb", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f1f3ee", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, color="#25313b")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, linestyle=":", color="#6b7280")

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="#6b7280",
        alpha=0.55,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    agency_count = sample.shape[1] - 1
    agency_colors = plt.cm.tab10(np.linspace(0, 1, max(agency_count, 1)))

    for channel in range(agency_count):
        track = valid_track(sample[:, channel, :])
        if len(track) == 0:
            continue
        ax.plot(
            track[:, 1],
            track[:, 0],
            marker="o",
            markersize=4,
            linewidth=1.2,
            alpha=0.82,
            color=agency_colors[channel],
            label=f"Agency {channel + 1}",
            transform=ccrs.PlateCarree(),
        )

    pred_track = valid_track(sample[:, -1, :])
    if len(pred_track) > 0:
        ax.plot(
            pred_track[:, 1],
            pred_track[:, 0],
            marker="*",
            markersize=9,
            linewidth=2.8,
            color="#c1121f",
            label="AI prediction",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        add_lead_labels(ax, pred_track, "#8f0d17")

    if gt_track is not None and len(gt_track) > 0:
        ax.plot(
            gt_track[:, 1],
            gt_track[:, 0],
            marker="s",
            markersize=5.5,
            linewidth=2.6,
            color="#111827",
            label="GT",
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        add_lead_labels(ax, gt_track, "#111827")

    ax.set_title(
        "Typhoon Ensemble Agencies, AI Prediction and GT\n"
        "20260051  Base: 2026-06-21 12:00 UTC / 2026-06-21 20:00 BJT",
        fontsize=13,
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=True, framealpha=0.92)
    plt.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot agency/member tracks and AI prediction from all_samples npy with Cartopy."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--base-time-bjt", default="2026-06-21 20:00:00")
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    all_samples = np.load(args.input)
    if all_samples.ndim != 4:
        raise ValueError(f"Expected npy shape (sample, lead, channel, 2), got {all_samples.shape}")
    if not 0 <= args.sample_index < all_samples.shape[0]:
        raise IndexError(f"sample-index {args.sample_index} out of range 0..{all_samples.shape[0] - 1}")

    gt_track = parse_gt_track(args.gt, args.base_time_bjt) if args.gt else None
    plot_sample(all_samples, args.sample_index, args.output, gt_track)
    print(f"Loaded: {args.input}")
    print(f"Shape:  {all_samples.shape}")
    if gt_track is not None:
        print(f"GT:     {args.gt}")
        print(f"GT lead-time points: {len(gt_track)}")
    print(f"Saved:  {args.output}")


if __name__ == "__main__":
    main()
