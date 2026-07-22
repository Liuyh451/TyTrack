import argparse
import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter


INSTITUTION_RE = re.compile(r"^机构:\s*(.+?)(?:\s+\(无数据\))?$")
POINT_RE = re.compile(
    r"^\s*\+\s*(?P<lead>\d+)h\s*\|\s*"
    r"Lat:\s*(?P<lat>-?\d+(?:\.\d+)?)\s*\|\s*"
    r"Lon:\s*(?P<lon>-?\d+(?:\.\d+)?)"
)

COLORS = {
    "中国台湾": "#1677b3",
    "中国香港": "#e67e22",
    "中国": "#2e8b57",
    "日本": "#c84343",
}

ENGLISH_NAMES = {
    "中国台湾": "Taiwan",
    "中国香港": "Hong Kong",
    "中国": "China",
    "日本": "Japan",
}

LABEL_OFFSETS = {
    ("中国台湾", 6): (-28, 6),
    ("中国台湾", 12): (-28, 6),
    ("中国台湾", 18): (-28, 6),
    ("中国台湾", 24): (-30, 5),
    ("中国台湾", 36): (-28, 6),
    ("中国台湾", 48): (-38, 6),
    ("中国台湾", 72): (-30, 6),
    ("中国香港", 24): (-28, 6),
    ("中国香港", 48): (-30, 6),
    ("中国香港", 72): (-28, 6),
    ("中国", 12): (-4, 14),
    ("中国", 24): (-7, -18),
    ("中国", 36): (8, -14),
    ("中国", 48): (10, -15),
    ("中国", 60): (-28, 8),
    ("中国", 72): (8, -14),
    ("日本", 24): (12, 10),
    ("日本", 48): (8, -16),
}

OURS_LABEL_OFFSETS = {
    0: (8, -1),
    24: (8, -15),
    48: (8, 7),
    72: (8, 7),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot typhoon forecast tracks with Cartopy.")
    parser.add_argument("forecast_records", type=Path)
    parser.add_argument("extra_track", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title", default="台风路径对比")
    return parser.parse_args()


def read_forecast_tracks(path):
    tracks = {}
    current_institution = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        institution_match = INSTITUTION_RE.match(raw_line)
        if institution_match:
            current_institution = institution_match.group(1)
            tracks.setdefault(current_institution, [])
            continue

        point_match = POINT_RE.match(raw_line)
        if point_match and current_institution:
            tracks[current_institution].append(
                (
                    int(point_match.group("lead")),
                    float(point_match.group("lon")),
                    float(point_match.group("lat")),
                )
            )

    return {name: points for name, points in tracks.items() if points}


def read_extra_track(path):
    points = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split()
        if len(fields) != 3:
            raise ValueError(
                f"{path}:{line_number}: expected 'lead_hour longitude latitude'"
            )
        lead, lon, lat = fields
        points.append((int(lead), float(lon), float(lat)))

    if not points:
        raise ValueError(f"no points found in {path}")
    return points


def padded_extent(points, padding=1.5):
    longitudes = [point[1] for point in points]
    latitudes = [point[2] for point in points]
    return [
        min(longitudes) - padding,
        max(longitudes) + padding,
        min(latitudes) - padding,
        max(latitudes) + padding,
    ]


def plot_tracks(forecast_tracks, extra_track, output, title):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    all_points = [point for points in forecast_tracks.values() for point in points]
    all_points.extend(extra_track)
    extent = padded_extent(all_points)
    ax.set_extent(extent, crs=projection)

    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#eaf4f8", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f1efe9", zorder=0)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#888888", linewidth=0.5)
    ax.coastlines(resolution="50m", color="#555555", linewidth=0.8)

    gridlines = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.6,
        color="#8c99a3",
        alpha=0.55,
        linestyle="--",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xformatter = LongitudeFormatter()
    gridlines.yformatter = LatitudeFormatter()
    gridlines.xlabel_style = {"size": 9}
    gridlines.ylabel_style = {"size": 9}

    for index, (institution, points) in enumerate(forecast_tracks.items()):
        longitudes = [point[1] for point in points]
        latitudes = [point[2] for point in points]
        color = COLORS.get(institution, f"C{index}")
        ax.plot(
            longitudes,
            latitudes,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.5,
            label=ENGLISH_NAMES.get(institution, institution),
            transform=projection,
            zorder=3,
        )

        for lead, lon, lat in points:
            offset = LABEL_OFFSETS.get((institution, lead), (6, 6))
            label = ax.annotate(
                f"{lead}h",
                xy=(lon, lat),
                xytext=offset,
                textcoords="offset points",
                fontsize=7.5,
                color=color,
                transform=projection,
                zorder=4,
            )
            label.set_path_effects(
                [path_effects.withStroke(linewidth=2.2, foreground="white")]
            )

    extra_longitudes = [point[1] for point in extra_track]
    extra_latitudes = [point[2] for point in extra_track]
    ax.plot(
        extra_longitudes,
        extra_latitudes,
        color="#111111",
        linewidth=2.8,
        marker="D",
        markerfacecolor="#ffd43b",
        markeredgecolor="#111111",
        markersize=6,
        label="Ours",
        transform=projection,
        zorder=5,
    )

    for lead, lon, lat in extra_track:
        label = ax.annotate(
            f"{lead}h",
            xy=(lon, lat),
            xytext=OURS_LABEL_OFFSETS.get(lead, (8, 7)),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#111111",
            transform=projection,
            zorder=6,
        )
        label.set_path_effects(
            [path_effects.withStroke(linewidth=2.5, foreground="white")]
        )

    ax.set_title(title, fontsize=16, pad=14)
    ax.text(
        0.01,
        0.01,
        "起报时间（北京时间）：2026-07-13 08:00",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 4},
        zorder=7,
    )
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#aaaaaa",
        fontsize=9,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    forecast_tracks = read_forecast_tracks(args.forecast_records)
    extra_track = read_extra_track(args.extra_track)
    plot_tracks(forecast_tracks, extra_track, args.output, args.title)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
