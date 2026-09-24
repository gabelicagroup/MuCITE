"""简易 Tkinter UI：用于检查 SIMION / Fluent 场的网格对齐并启动离线烘焙。"""

from __future__ import annotations

import queue
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

try:
    from PIL import Image, ImageTk
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    Image = None
    ImageTk = None

from .field_baker_layout import build_field_baker_layout


class FieldBakerUI:
    """一个面向人工检查的小型场对齐 UI。

    功能目标：
    - 选择 SIMION / Fluent 输入文件
    - 设置 z 偏移和背景参数
    - 一键启动场烘焙
    - 直接在窗口中查看诊断图，快速判断 capillary 出口是否对齐
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Simu_IonSource Field Baker UI")
        self.root.geometry("1380x860")

        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.running = False
        self._preview_refs = []

        self.simion_dc_var = tk.StringVar()
        self.simion_rf_var = tk.StringVar()
        self.fluent_var = tk.StringVar()
        self.output_npy_var = tk.StringVar(value=str(Path("outputs") / "baked_fields.npy"))
        self.output_plot_dir_var = tk.StringVar(value=str(Path("outputs") / "field_baker_plots"))

        self.offset_simion_var = tk.StringVar(value="0.0")
        self.offset_fluent_var = tk.StringVar(value="-95.0")
        self.fluent_capillary_total_length_var = tk.StringVar(value="101.0")
        self.fluent_capillary_radius_var = tk.StringVar(value="0.25")
        self.fluent_capillary_external_start_z_var = tk.StringVar(value="4.5")
        self.background_pressure_var = tk.StringVar(value="200.0")
        self.background_temperature_var = tk.StringVar(value="300.0")
        self.phi_key_var = tk.StringVar(value="phi_dc_v")
        self.simion_pa_grids_var = tk.StringVar(value="")
        self.simion_dc_voltage_scale_var = tk.StringVar(value="1.0")
        self.z_min_var = tk.StringVar(value="0.0")
        self.z_max_var = tk.StringVar(value="50.0")
        self.r_min_var = tk.StringVar(value="0.0")
        self.r_max_var = tk.StringVar(value="10.0")
        self.dz_var = tk.StringVar(value="0.1")
        self.dr_var = tk.StringVar(value="0.1")
        self.capillary_exit_z_var = tk.StringVar(value="6.0")
        self.clip_fluent_after_z_var = tk.StringVar(value="")

        self.status_var = tk.StringVar(value="Ready.")

        self._build_layout()

    def _build_layout(self) -> None:
        build_field_baker_layout(self)

    def _append_log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _add_file_row(self, parent: ttk.LabelFrame, label: str, var: tk.StringVar, row: int, *, save_mode: bool = False) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=var, width=42)
        entry.grid(row=row, column=1, sticky="ew", padx=(10, 8), pady=4)
        button = ttk.Button(
            parent,
            text="Browse",
            command=lambda current_var=var, save=save_mode: self._browse_file(current_var, save_mode=save),
        )
        button.grid(row=row, column=2, pady=4)
        parent.columnconfigure(1, weight=1)

    def _add_directory_row(self, parent: ttk.LabelFrame, label: str, var: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=var, width=42)
        entry.grid(row=row, column=1, sticky="ew", padx=(10, 8), pady=4)
        button = ttk.Button(parent, text="Browse", command=lambda current_var=var: self._browse_directory(current_var))
        button.grid(row=row, column=2, pady=4)
        parent.columnconfigure(1, weight=1)

    def _add_entry_row(self, parent: ttk.LabelFrame, label: str, var: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=var, width=18).grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=4)
        parent.columnconfigure(1, weight=1)

    def _browse_file(self, var: tk.StringVar, *, save_mode: bool = False) -> None:
        if save_mode:
            path = filedialog.asksaveasfilename(
                title="Select output file",
                defaultextension=".npy",
                filetypes=[("NumPy file", "*.npy"), ("All files", "*.*")],
            )
        else:
            path = filedialog.askopenfilename(
                title="Select field file",
                filetypes=[("Supported field files", "*.csv;*.patxt"), ("CSV file", "*.csv"), ("SIMION PA Text", "*.patxt"), ("All files", "*.*")],
            )
        if path:
            var.set(path)

    def _browse_directory(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(title="Select directory")
        if path:
            var.set(path)

    @staticmethod
    def _parse_optional_float(value: str):
        value = value.strip()
        return None if not value else float(value)

    def _build_config(self) -> Any:
        from ...data.tools.field_baker import FieldBakeConfig

        try:
            return FieldBakeConfig(
                simion_dc_csv=Path(self.simion_dc_var.get()) if self.simion_dc_var.get().strip() else None,
                simion_rf_csv=Path(self.simion_rf_var.get()) if self.simion_rf_var.get().strip() else None,
                fluent_csv=Path(self.fluent_var.get()) if self.fluent_var.get().strip() else None,
                offset_z_simion_mm=float(self.offset_simion_var.get()),
                offset_z_fluent_mm=float(self.offset_fluent_var.get()),
                output_npy=Path(self.output_npy_var.get().strip() or str(Path("outputs") / "baked_fields.npy")),
                output_plot_dir=Path(self.output_plot_dir_var.get().strip() or str(Path("outputs") / "field_baker_plots")),
                background_pressure_pa=float(self.background_pressure_var.get()),
                background_temperature_k=float(self.background_temperature_var.get()),
                phi_key_for_plot=self.phi_key_var.get(),
                z_min_mm=float(self.z_min_var.get()),
                z_max_mm=float(self.z_max_var.get()),
                r_min_mm=float(self.r_min_var.get()),
                r_max_mm=float(self.r_max_var.get()),
                dz_mm=float(self.dz_var.get()),
                dr_mm=float(self.dr_var.get()),
                capillary_exit_z_mm=float(self.capillary_exit_z_var.get()),
                fluent_capillary_total_length_mm=float(
                    self.fluent_capillary_total_length_var.get()
                ),
                fluent_capillary_radius_mm=float(
                    self.fluent_capillary_radius_var.get()
                ),
                fluent_capillary_external_start_z_mm=float(
                    self.fluent_capillary_external_start_z_var.get()
                ),
                clip_fluent_after_z_mm=self._parse_optional_float(self.clip_fluent_after_z_var.get()),
                simion_pa_effective_grids_per_mm=self._parse_optional_float(
                    self.simion_pa_grids_var.get()
                ),
                simion_dc_voltage_scale=float(self.simion_dc_voltage_scale_var.get()),
            )
        except ValueError as exc:
            raise ValueError(f"数值参数格式错误：{exc}") from exc

    def _reset_grid_defaults(self) -> None:
        self.z_min_var.set("0.0")
        self.z_max_var.set("50.0")
        self.r_min_var.set("0.0")
        self.r_max_var.set("10.0")
        self.dz_var.set("0.1")
        self.dr_var.set("0.1")
        self.capillary_exit_z_var.set("6.0")
        self.fluent_capillary_total_length_var.set("101.0")
        self.fluent_capillary_radius_var.set("0.25")
        self.fluent_capillary_external_start_z_var.set("4.5")
        self.clip_fluent_after_z_var.set("")
        self.simion_pa_grids_var.set("10.0")
        self.simion_dc_voltage_scale_var.set("1.0")
        self._append_log("Grid parameters reset to defaults: z=[0, 50] mm, r=[0, 10] mm, dz=dr=0.1 mm.")

    def _start_bake(self) -> None:
        if self.running:
            return

        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid Parameters", str(exc))
            return

        self.running = True
        self.run_button.configure(state=tk.DISABLED)
        self.status_var.set("Baking fields...")
        self._append_log("Starting field baking workflow...")
        worker = threading.Thread(target=self._worker, args=(config,), daemon=True)
        worker.start()
        self.root.after(100, self._poll_queue)

    def _worker(self, config: Any) -> None:
        from ...data.tools.field_baker import run_field_bake

        try:
            result = run_field_bake(config)
            self.message_queue.put(("result", result))
        except Exception:
            self.message_queue.put(("error", traceback.format_exc()))

    def _load_preview(self, label_widget: ttk.Label, image_path: Path, *, max_size: tuple[int, int]) -> None:
        if Image is None or ImageTk is None:
            label_widget.configure(text=f"Preview unavailable (Pillow not installed).\n{image_path}", image="")
            return

        image = Image.open(image_path)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label_widget.configure(image=photo, text="")
        self._preview_refs.append(photo)

    def _handle_result(self, result: object) -> None:
        payload = dict(result)  # type: ignore[arg-type]
        plot_paths = [Path(path) for path in payload.get("plot_paths", [])]
        output_npy = payload.get("output_npy")
        grid_metadata = dict(payload.get("grid_metadata", {}))
        simion_summary = dict(payload.get("simion_summary", {}))

        if grid_metadata:
            self._append_log(
                "Resolved global grid: "
                f"r=[{grid_metadata['r_min_m'] * 1.0e3:.3f}, {grid_metadata['r_max_m'] * 1.0e3:.3f}] mm, "
                f"z=[{grid_metadata['z_min_m'] * 1.0e3:.3f}, {grid_metadata['z_max_m'] * 1.0e3:.3f}] mm, "
                f"dr={grid_metadata['dr_m'] * 1.0e3:.3f} mm, dz={grid_metadata['dz_m'] * 1.0e3:.3f} mm, "
                f"shape=({grid_metadata['nr']}, {grid_metadata['nz']})"
            )
        if simion_summary.get("dc_source_format") == "patxt":
            self._append_log(
                "SIMION DC source: "
                f"patxt, ng(header)={simion_summary.get('dc_pa_header_ng')}, "
                f"effective grids/mm={simion_summary.get('dc_pa_effective_grids_per_mm')}, "
                f"voltage_scale={simion_summary.get('dc_voltage_scale')}, "
                f"extent z=[{simion_summary.get('dc_source_z_min_m', 0.0) * 1.0e3:.3f}, {simion_summary.get('dc_source_z_max_m', 0.0) * 1.0e3:.3f}] mm, "
                f"r=[{simion_summary.get('dc_source_r_min_m', 0.0) * 1.0e3:.3f}, {simion_summary.get('dc_source_r_max_m', 0.0) * 1.0e3:.3f}] mm, "
                f"inside_global={simion_summary.get('dc_source_fully_inside_global_grid')}"
            )
        if simion_summary.get("rf_source_format") == "patxt":
            self._append_log(
                "SIMION RF source: "
                f"patxt, ng(header)={simion_summary.get('rf_pa_header_ng')}, "
                f"effective grids/mm={simion_summary.get('rf_pa_effective_grids_per_mm')}, "
                f"Vref={simion_summary.get('rf_reference_peak_voltage_v')} V, "
                f"normalization={simion_summary.get('rf_normalization')}, "
                f"extent z=[{simion_summary.get('rf_source_z_min_m', 0.0) * 1.0e3:.3f}, {simion_summary.get('rf_source_z_max_m', 0.0) * 1.0e3:.3f}] mm, "
                f"r=[{simion_summary.get('rf_source_r_min_m', 0.0) * 1.0e3:.3f}, {simion_summary.get('rf_source_r_max_m', 0.0) * 1.0e3:.3f}] mm, "
                f"inside_global={simion_summary.get('rf_source_fully_inside_global_grid')}"
            )

        self._append_log(f"Saved baked field file: {output_npy}")
        for path in plot_paths:
            self._append_log(f"Saved sanity plot: {path}")

        if len(plot_paths) >= 2:
            self._preview_refs = []
            self._load_preview(self.heatmap_label, plot_paths[0], max_size=(840, 620))
            self._load_preview(self.centerline_label, plot_paths[1], max_size=(840, 620))

        self.status_var.set("Completed. Review the preview figures on the right.")
        self.running = False
        self.run_button.configure(state=tk.NORMAL)

    def _handle_error(self, error_text: str) -> None:
        self._append_log("Error:")
        self._append_log(error_text)
        self.status_var.set("Failed. See log for traceback.")
        self.running = False
        self.run_button.configure(state=tk.NORMAL)
        messagebox.showerror("Field Baker Error", error_text)

    def _poll_queue(self) -> None:
        while True:
            try:
                message_type, payload = self.message_queue.get_nowait()
            except queue.Empty:
                break

            if message_type == "result":
                self._handle_result(payload)
            elif message_type == "error":
                self._handle_error(str(payload))

        if self.running:
            self.root.after(100, self._poll_queue)

    def run(self) -> None:
        self.root.mainloop()


def launch_field_baker_ui() -> None:
    FieldBakerUI().run()


if __name__ == "__main__":
    launch_field_baker_ui()
