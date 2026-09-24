"""Offline plotting utilities for exported macro-step particle snapshots."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Snapshot:
    macro_step: int
    time_s: float
    particle_count: int
    data: np.ndarray


def _load_index(index_path: Path) -> list[dict[str, str]]:
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_metadata(metadata_path: Path) -> dict[str, object]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_snapshots(input_path: Path) -> tuple[list[Snapshot], dict[str, object]]:
    input_path = Path(input_path)

    if input_path.is_dir():
        metadata = _load_metadata(input_path / "metadata.json")
        rows = _load_index(input_path / "snapshots_index.csv")
        snapshots: list[Snapshot] = []
        if metadata.get("file_format") == "h5":
            try:
                import h5py  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError("Reading HDF5 exports requires h5py.") from exc

            with h5py.File(input_path / "snapshots.h5", "r") as handle:
                for row in rows:
                    data = np.asarray(handle[row["storage_ref"]][...], dtype=np.float64)
                    snapshots.append(
                        Snapshot(
                            macro_step=int(row["macro_step"]),
                            time_s=float(row["time_s"]),
                            particle_count=int(row["particle_count"]),
                            data=data,
                        )
                    )
        else:
            for row in rows:
                data = np.load(input_path / row["storage_ref"])
                snapshots.append(
                    Snapshot(
                        macro_step=int(row["macro_step"]),
                        time_s=float(row["time_s"]),
                        particle_count=int(row["particle_count"]),
                        data=np.asarray(data, dtype=np.float64),
                    )
                )
        return snapshots, metadata

    if input_path.suffix.lower() == ".h5":
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("Reading HDF5 exports requires h5py.") from exc

        metadata = {}
        snapshots: list[Snapshot] = []
        with h5py.File(input_path, "r") as handle:
            metadata["columns"] = json.loads(handle.attrs["columns"])
            metadata["units"] = json.loads(handle.attrs["units"])
            group = handle["snapshots"]
            for dataset_name in sorted(group.keys()):
                dataset = group[dataset_name]
                snapshots.append(
                    Snapshot(
                        macro_step=int(dataset.attrs["macro_step"]),
                        time_s=float(dataset.attrs["time_s"]),
                        particle_count=int(dataset.attrs["particle_count"]),
                        data=np.asarray(dataset[...], dtype=np.float64),
                    )
                )
        return snapshots, metadata

    raise ValueError("input_path must be an export directory or an HDF5 snapshot file.")


def plot_phase_space(snapshots: list[Snapshot], output_path: Path) -> None:
    if not snapshots:
        raise ValueError("No snapshots available for plotting.")

    all_points = [snapshot.data[:, :2] for snapshot in snapshots if snapshot.data.size > 0]
    if not all_points:
        raise ValueError("All snapshots are empty.")

    aggregate = np.vstack(all_points)
    final_snapshot = next((snapshot for snapshot in reversed(snapshots) if snapshot.data.size > 0), snapshots[-1])

    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=160)
    ax.scatter(aggregate[:, 1] * 1.0e3, aggregate[:, 0] * 1.0e3, s=4, alpha=0.12, color="#1f77b4", label="All snapshots")

    if final_snapshot.data.size > 0:
        ax.scatter(
            final_snapshot.data[:, 1] * 1.0e3,
            final_snapshot.data[:, 0] * 1.0e3,
            s=10,
            alpha=0.7,
            color="#d62728",
            label=f"Final snapshot (macro {final_snapshot.macro_step})",
        )

    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r [mm]")
    ax.set_title("Ion Trajectory Scatter / Beam Envelope")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_temperature_evolution(snapshots: list[Snapshot], output_path: Path) -> None:
    if not snapshots:
        raise ValueError("No snapshots available for plotting.")

    z_mean_mm: list[float] = []
    teff_mean_k: list[float] = []
    teff_std_k: list[float] = []

    for snapshot in snapshots:
        if snapshot.data.size == 0:
            continue

        z_positions_m = snapshot.data[:, 1]
        temperatures_k = snapshot.data[:, 2]
        z_mean_mm.append(float(np.mean(z_positions_m) * 1.0e3))
        teff_mean_k.append(float(np.mean(temperatures_k)))
        teff_std_k.append(float(np.std(temperatures_k)))

    fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=160)
    ax.errorbar(
        z_mean_mm,
        teff_mean_k,
        yerr=teff_std_k,
        fmt="-o",
        color="#ff7f0e",
        ecolor="#444444",
        elinewidth=1.0,
        capsize=3,
        markersize=4,
    )
    ax.set_xlabel("Mean z [mm]")
    ax.set_ylabel("Effective Temperature [K]")
    ax.set_title("Temperature Evolution Along the Beam (1σ Error Bar)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot exported Simu_IonSource particle snapshots.")
    parser.add_argument("input_path", help="Export directory or snapshots.h5 file.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for generated PNG files. Defaults to the export directory or the HDF5 file directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    snapshots, metadata = load_snapshots(input_path)
    del metadata

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif input_path.is_dir():
        output_dir = input_path
    else:
        output_dir = input_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_phase_space(snapshots, output_dir / "phase_space.png")
    plot_temperature_evolution(snapshots, output_dir / "temperature_evolution.png")

    print(f"Saved: {output_dir / 'phase_space.png'}")
    print(f"Saved: {output_dir / 'temperature_evolution.png'}")


if __name__ == "__main__":
    main()
