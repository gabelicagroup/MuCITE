"""Snapshot exporter implementing the core ``SnapshotSink`` port."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .snapshot_plots import write_spatial_plots, write_trajectory_plots


@dataclass(frozen=True)
class ExportRecord:
    """Metadata for one exported macro-step snapshot."""

    macro_step: int
    time_s: float
    particle_count: int
    storage_ref: str


class DataLogger:
    """Persist alive-ion spatial snapshots and representative trajectories.

    Snapshot columns keep the legacy first four fields stable:

    ``[r_m, z_m, teff_k, kinetic_energy_j]``.

    Newer runs append Cartesian position/velocity and particle accounting fields
    so the same export can be used for spatial plume plots and sampled trajectory
    reconstruction.

    Two storage modes are supported:

    - ``npy``: one ``.npy`` file per snapshot plus a CSV index,
    - ``h5``: one HDF5 container plus the same CSV index for quick inspection.
    """

    columns = (
        "r_m",
        "z_m",
        "teff_k",
        "kinetic_energy_j",
        "x_m",
        "y_m",
        "vx_m_per_s",
        "vy_m_per_s",
        "vz_m_per_s",
        "speed_m_per_s",
        "kinetic_energy_ev",
        "represented_real_ions",
        "parent_real_ions_remaining",
        "fragmented_real_ions_represented",
        "track_id",
        "slot_id",
        "status_code",
        "collision_count_per_ion",
    )

    units = {
        "r_m": "m",
        "z_m": "m",
        "teff_k": "K",
        "kinetic_energy_j": "J",
        "x_m": "m",
        "y_m": "m",
        "vx_m_per_s": "m/s",
        "vy_m_per_s": "m/s",
        "vz_m_per_s": "m/s",
        "speed_m_per_s": "m/s",
        "kinetic_energy_ev": "eV",
        "represented_real_ions": "real ions",
        "parent_real_ions_remaining": "real ions",
        "fragmented_real_ions_represented": "real ions",
        "track_id": "integer",
        "slot_id": "integer",
        "status_code": "integer",
        "collision_count_per_ion": "count",
    }

    def __init__(
        self,
        output_dir: Path,
        *,
        export_every_macro_steps: int = 1,
        file_format: str = "npy",
        representative_trajectory_count: int = 200,
        trajectory_record_every_snapshots: int = 1,
        make_plots: bool = True,
        plot_max_points: int = 200_000,
        trajectory_plot_max_tracks: int = 50,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.export_every_macro_steps = int(export_every_macro_steps)
        if self.export_every_macro_steps < 0:
            raise ValueError("export_every_macro_steps must be non-negative.")
        self.representative_trajectory_count = max(0, int(representative_trajectory_count))
        self.trajectory_record_every_snapshots = max(1, int(trajectory_record_every_snapshots))
        self.make_plots = bool(make_plots)
        self.plot_max_points = max(1, int(plot_max_points))
        self.trajectory_plot_max_tracks = max(
            0,
            int(trajectory_plot_max_tracks),
        )

        self.file_format = file_format.lower()
        if self.file_format not in {"npy", "h5"}:
            raise ValueError("file_format must be either 'npy' or 'h5'.")

        self.records: list[ExportRecord] = []
        self.last_logged_macro_step: Optional[int] = None
        self.generated_artifacts: list[Path] = []
        self._selected_track_ids: set[int] = set()
        self._trajectory_handle = None
        self._trajectory_writer: Optional[csv.writer] = None
        self._h5_file = None
        self._h5_path: Optional[Path] = None

        if self.file_format == "h5":
            self._open_h5_storage()

        if self.representative_trajectory_count > 0:
            self._open_trajectory_output()

        self._write_metadata()

    def _open_h5_storage(self) -> None:
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "HDF5 export requested but h5py is not installed."
            ) from exc
        self._h5_path = self.output_dir / "snapshots.h5"
        self._h5_file = h5py.File(self._h5_path, "w")
        self._h5_file.attrs["columns"] = json.dumps(self.columns)
        self._h5_file.attrs["units"] = json.dumps(self.units)

    def _open_trajectory_output(self) -> None:
        path = self.output_dir / "representative_trajectories.csv"
        self._trajectory_handle = path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self._trajectory_writer = csv.writer(self._trajectory_handle)
        self._trajectory_writer.writerow(
            ["macro_step", "time_s", *self.columns]
        )
        self.generated_artifacts.append(path)

    def _write_metadata(self) -> None:
        metadata = {
            "schema_version": 2,
            "columns": list(self.columns),
            "units": self.units,
            "file_format": self.file_format,
            "export_every_macro_steps": self.export_every_macro_steps,
            "representative_trajectory_count": self.representative_trajectory_count,
            "trajectory_record_every_snapshots": self.trajectory_record_every_snapshots,
            "make_plots": self.make_plots,
            "plot_max_points": self.plot_max_points,
            "trajectory_plot_max_tracks": self.trajectory_plot_max_tracks,
        }
        metadata_path = self.output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def should_log(self, macro_step: int) -> bool:
        return self.export_every_macro_steps > 0 and macro_step % self.export_every_macro_steps == 0

    def log_snapshot(self, macro_step: int, time_s: float, particle_snapshot: np.ndarray) -> None:
        data = np.asarray(particle_snapshot, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("particle_snapshot must be a 2D array.")
        if data.shape[1] == 4 and len(self.columns) > 4:
            padding = np.full((data.shape[0], len(self.columns) - 4), np.nan, dtype=np.float64)
            data = np.column_stack((data, padding))
        if data.shape[1] != len(self.columns):
            raise ValueError(f"particle_snapshot must have shape (N, {len(self.columns)}).")

        particle_count = int(data.shape[0])
        if self.file_format == "npy":
            file_name = f"snapshot_macro_{macro_step:05d}.npy"
            np.save(self.output_dir / file_name, data)
            storage_ref = file_name
        else:
            assert self._h5_file is not None
            dataset_name = f"snapshots/macro_{macro_step:05d}"
            dataset = self._h5_file.create_dataset(dataset_name, data=data, compression="gzip")
            dataset.attrs["macro_step"] = int(macro_step)
            dataset.attrs["time_s"] = float(time_s)
            dataset.attrs["particle_count"] = particle_count
            storage_ref = dataset_name

        self.records.append(
            ExportRecord(
                macro_step=int(macro_step),
                time_s=float(time_s),
                particle_count=particle_count,
                storage_ref=storage_ref,
            )
        )
        self.last_logged_macro_step = int(macro_step)
        self._record_representative_trajectories(int(macro_step), float(time_s), data)

    def _column_index(self, name: str) -> int:
        return self.columns.index(name)

    def _record_representative_trajectories(self, macro_step: int, time_s: float, data: np.ndarray) -> None:
        if self._trajectory_writer is None or data.size == 0:
            return
        snapshot_index = len(self.records) - 1
        if snapshot_index % self.trajectory_record_every_snapshots != 0:
            return
        track_col = self._column_index("track_id")
        track_values = data[:, track_col]
        valid_tracks = track_values[np.isfinite(track_values) & (track_values >= 0.0)].astype(np.int64)
        if valid_tracks.size == 0:
            return
        if not self._selected_track_ids:
            unique_tracks = np.unique(valid_tracks)
            if unique_tracks.size <= self.representative_trajectory_count:
                selected = unique_tracks
            else:
                offsets = np.linspace(0, unique_tracks.size - 1, self.representative_trajectory_count, dtype=np.int64)
                selected = unique_tracks[offsets]
            self._selected_track_ids = {int(value) for value in selected.tolist()}

        selected_mask = np.isin(valid_tracks, list(self._selected_track_ids))
        if not np.any(selected_mask):
            return
        finite_mask = np.isfinite(track_values) & (track_values >= 0.0)
        data_mask = np.zeros(track_values.shape, dtype=bool)
        data_mask[finite_mask] = np.isin(track_values[finite_mask].astype(np.int64), list(self._selected_track_ids))
        for row in data[data_mask]:
            self._trajectory_writer.writerow([macro_step, f"{time_s:.12e}", *[f"{value:.12e}" for value in row]])

    def _iter_snapshot_arrays(self) -> Iterable[tuple[ExportRecord, np.ndarray]]:
        if self.file_format == "h5":
            if self._h5_path is None:
                return
            try:
                import h5py  # type: ignore
            except ModuleNotFoundError:
                return
            with h5py.File(self._h5_path, "r") as handle:
                for record in self.records:
                    yield record, np.asarray(handle[record.storage_ref][...], dtype=np.float64)
        else:
            for record in self.records:
                path = self.output_dir / record.storage_ref
                if path.exists():
                    yield record, np.asarray(np.load(path), dtype=np.float64)

    def _sample_plot_points(self) -> tuple[np.ndarray, np.ndarray]:
        nonempty = [record for record in self.records if record.particle_count > 0]
        if not nonempty:
            return np.zeros((0, len(self.columns)), dtype=np.float64), np.zeros(0, dtype=np.float64)
        per_snapshot = max(1, self.plot_max_points // max(len(nonempty), 1))
        point_blocks: list[np.ndarray] = []
        time_blocks: list[np.ndarray] = []
        for record, snapshot in self._iter_snapshot_arrays():
            if snapshot.size == 0:
                continue
            take = min(snapshot.shape[0], per_snapshot)
            if take < snapshot.shape[0]:
                offsets = np.linspace(0, snapshot.shape[0] - 1, take, dtype=np.int64)
                snapshot = snapshot[offsets]
            point_blocks.append(snapshot)
            time_blocks.append(np.full(snapshot.shape[0], record.time_s, dtype=np.float64))
        if not point_blocks:
            return np.zeros((0, len(self.columns)), dtype=np.float64), np.zeros(0, dtype=np.float64)
        return np.vstack(point_blocks), np.concatenate(time_blocks)

    def _write_spatial_plots(self) -> None:
        write_spatial_plots(self)

    def _write_trajectory_plots(self) -> None:
        write_trajectory_plots(self)

    def close(self) -> None:
        index_path = self.output_dir / "snapshots_index.csv"
        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["macro_step", "time_s", "particle_count", "storage_ref"])
            for record in self.records:
                writer.writerow([record.macro_step, f"{record.time_s:.12e}", record.particle_count, record.storage_ref])

        if self._h5_file is not None:
            self._h5_file.flush()
            self._h5_file.close()
            self._h5_file = None
        if self._trajectory_handle is not None:
            self._trajectory_handle.flush()
            self._trajectory_handle.close()
            self._trajectory_handle = None
        self._write_spatial_plots()
        self._write_trajectory_plots()
