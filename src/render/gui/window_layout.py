"""Main-window layout and visualization widget construction."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .plotting import make_empty_figure


PLOT_TABS = (
    "Snapshot",
    "XY",
    "RZ",
    "Phase Space",
    "Temperature",
    "Energy",
    "Terminal Current",
    "Field",
    "Gas Flow",
    "Alignment",
    "Loss Map",
)


class ScrollableFrame(ttk.Frame):
    """A vertical scroll wrapper for dense control panels."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, width=330)
        self.scrollbar = ttk.Scrollbar(
            self,
            orient=tk.VERTICAL,
            command=self.canvas.yview,
        )
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window(
            (0, 0),
            window=self.inner,
            anchor="nw",
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)


class _WindowLayoutMixin:
    def _build_layout(self) -> None:
        self._build_menu_bar()
        root = ttk.Frame(self.root, padding=8)
        root.pack(fill=tk.BOTH, expand=True)
        self._build_toolbar(root)
        self._build_project_panel(root)
        vertical = ttk.PanedWindow(root, orient=tk.VERTICAL)
        vertical.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        main_pane = ttk.PanedWindow(vertical, orient=tk.HORIZONTAL)
        vertical.add(main_pane, weight=5)
        controls = ScrollableFrame(main_pane)
        main_pane.add(controls, weight=0)
        self._build_control_panel(controls.inner)
        right = ttk.Frame(main_pane)
        main_pane.add(right, weight=1)
        self._build_visualization_panel(right)
        bottom = ttk.Frame(vertical)
        vertical.add(bottom, weight=1)
        self._build_console(bottom)

    def _build_menu_bar(self) -> None:
        menu_bar = tk.Menu(self.root)
        project = tk.Menu(menu_bar, tearoff=False)
        project.add_command(label="New", command=self._new_session)
        project.add_command(label="Load", command=self._load_config)
        project.add_command(label="Save", command=self._save_config)
        project.add_separator()
        project.add_command(label="Exit", command=self._on_close)
        menu_bar.add_cascade(label="Project", menu=project)
        config = tk.Menu(menu_bar, tearoff=False)
        config.add_command(label="Beam", command=self._open_beam_dialog)
        config.add_command(label="Field", command=self._open_field_dialog)
        config.add_command(label="PIC", command=self._open_pic_dialog)
        config.add_command(label="Collision", command=self._open_collision_dialog)
        menu_bar.add_cascade(label="Config", menu=config)
        help_menu = tk.Menu(menu_bar, tearoff=False)
        help_menu.add_command(label="Parameter Reference", command=self._show_parameter_reference)
        help_menu.add_command(label="Workflow", command=self._show_workflow_help)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu_bar)

    def _build_toolbar(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X)
        ttk.Label(
            toolbar,
            text="MuCITE GUI",
            font=("Segoe UI", 12, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 16))
        signature = ttk.Label(toolbar, text="Dr. Yihui Yan", foreground="#6b7280")
        signature.pack(side=tk.RIGHT, padx=(16, 2))

    def _build_project_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Project", padding=8)
        panel.pack(fill=tk.X, pady=(8, 0))
        panel.columnconfigure(0, weight=1)
        fields = ttk.Frame(panel)
        fields.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        fields.columnconfigure(1, weight=1)
        fields.columnconfigure(4, weight=1)
        actions = ttk.Frame(panel)
        actions.grid(row=0, column=1, sticky="nsew")
        self._build_project_header(fields)
        self._build_project_paths(fields)
        self._build_runtime_summary_row(fields)
        self._build_project_actions(actions)

    def _build_project_actions(self, parent: ttk.Frame) -> None:
        actions = (
            ("Go", self._start_simulation),
            ("Stop", self._stop_current_task),
            ("Generate Report", self._start_report),
        )
        parent.columnconfigure(0, weight=1)
        for row, (text, command) in enumerate(actions):
            parent.rowconfigure(row, weight=1)
            ttk.Button(
                parent,
                text=text,
                command=command,
                padding=(18, 10),
                width=18,
            ).grid(row=row, column=0, sticky="nsew", pady=2)

    def _build_project_header(self, panel: ttk.Frame) -> None:
        ttk.Label(panel, text="Project").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(panel, textvariable=self.session_name_var).grid(
            row=0, column=1, sticky="ew", pady=3
        )
        ttk.Label(panel, text="Backend").grid(
            row=0, column=2, sticky="w", padx=(12, 6), pady=3
        )
        ttk.Combobox(
            panel,
            textvariable=self.backend_var,
            values=("cpu", "taichi"),
            state="readonly",
            width=10,
        ).grid(row=0, column=3, sticky="w", pady=3)
        ttk.Label(panel, text="Status").grid(
            row=0, column=4, sticky="e", padx=(12, 6), pady=3
        )
        ttk.Label(
            panel,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=5, sticky="w", pady=3)

    def _build_project_paths(self, panel: ttk.Frame) -> None:
        ttk.Label(panel, text="Output dir").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(panel, textvariable=self.output_dir_var).grid(
            row=1, column=1, columnspan=4, sticky="ew", pady=3
        )
        ttk.Button(
            panel,
            text="Browse",
            command=self._browse_output_dir,
        ).grid(row=1, column=5, sticky="ew", padx=(8, 0), pady=3)
        ttk.Label(panel, text="Baked field").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=3
        )
        ttk.Entry(panel, textvariable=self.static_field_var).grid(
            row=2, column=1, columnspan=4, sticky="ew", pady=3
        )
        ttk.Button(
            panel,
            text="Browse",
            command=self._browse_static_field,
        ).grid(row=2, column=5, sticky="ew", padx=(8, 0), pady=3)

    def _build_runtime_summary_row(self, panel: ttk.Frame) -> None:
        summary = ttk.Frame(panel)
        summary.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(6, 0))
        values = (
            ("Macro", self.macro_var),
            ("Time", self.time_var),
            ("Active weighted ion packs", self.alive_var),
            ("Collisions", self.collisions_var),
            ("Exit current", self.exit_current_var),
            ("Exit real ions/bin", self.exit_ions_var),
        )
        for index, (label, var) in enumerate(values):
            ttk.Label(summary, text=f"{label}:").grid(
                row=0,
                column=index * 2,
                sticky="w",
                padx=(0 if index == 0 else 12, 4),
            )
            ttk.Label(summary, textvariable=var).grid(
                row=0,
                column=index * 2 + 1,
                sticky="w",
            )

    def _build_control_panel(self, parent: ttk.Frame) -> None:
        self._build_beam_group(parent)
        self._build_field_group(parent)
        self._build_runtime_group(parent)
        self._build_analysis_group(parent)
        self._build_snapshot_group(parent)

    def _build_beam_group(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Beam Setup", padding=8)
        group.pack(fill=tk.X, pady=(0, 8), padx=4)
        ttk.Button(
            group,
            text="Open Beam Dialog",
            command=self._open_beam_dialog,
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            group,
            text="Preview / Smoke Test",
            command=self._start_beam_smoke,
        ).pack(fill=tk.X, pady=2)
        ttk.Label(
            group,
            textvariable=self.beam_summary_var,
            wraplength=300,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(6, 0))

    def _build_field_group(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Field Baker", padding=8)
        group.pack(fill=tk.X, pady=(0, 8), padx=4)
        buttons = (
            ("Open Field Baker Dialog", self._open_field_dialog),
            ("Run Bake", self._start_field_bake),
            ("Load Existing Baked Field", self._browse_static_field),
            ("Show Diagnostics", self._show_field_diagnostics),
        )
        for text, command in buttons:
            ttk.Button(group, text=text, command=command).pack(
                fill=tk.X,
                pady=2,
            )
        ttk.Label(
            group,
            textvariable=self.field_summary_var,
            wraplength=300,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(6, 0))

    def _build_analysis_group(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Analysis", padding=8)
        group.pack(fill=tk.X, pady=(0, 8), padx=4)
        buttons = (
            ("Browse Snapshots", self._load_snapshot_index),
            ("Plot Summary", self._plot_last_summary),
            ("Export Current Figure", self._export_current_figure),
            ("Open Output Directory", self._open_output_dir),
        )
        for text, command in buttons:
            ttk.Button(group, text=text, command=command).pack(
                fill=tk.X,
                pady=2,
            )

    def _build_snapshot_group(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Snapshot Browser", padding=8)
        group.pack(fill=tk.BOTH, expand=True, pady=(0, 8), padx=4)
        self.snapshot_listbox = tk.Listbox(group, height=9)
        self.snapshot_listbox.pack(fill=tk.BOTH, expand=True)
        self.snapshot_listbox.bind(
            "<<ListboxSelect>>",
            self._on_snapshot_selected,
        )

    def _build_visualization_panel(self, parent: ttk.Frame) -> None:
        self.plot_notebook = ttk.Notebook(parent)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True)
        for name in PLOT_TABS:
            frame = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(frame, text=name)
            self.plot_frames[name] = frame
        self.plot_notebook.bind("<<NotebookTabChanged>>", self._on_plot_tab_changed)
        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            controls,
            text="Prev",
            command=lambda: self._step_snapshot(-1),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            controls,
            text="Next",
            command=lambda: self._step_snapshot(1),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(
            controls,
            text="Auto-refresh",
            variable=self.auto_refresh_var,
        ).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Button(
            controls,
            text="Export Figure",
            command=self._export_current_figure,
        ).pack(side=tk.RIGHT, padx=2)

    def _build_console(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        for name in ("Logs", "Warnings", "Validation", "Run Summary"):
            frame = ttk.Frame(notebook)
            text = scrolledtext.ScrolledText(
                frame,
                wrap=tk.WORD,
                height=8,
                font=("Consolas", 9),
            )
            text.pack(fill=tk.BOTH, expand=True)
            text.configure(state=tk.DISABLED)
            notebook.add(frame, text=name)
            self.console_tabs[name] = text
        self._build_terminal_console(notebook)

    def _initialize_plot_tabs(self) -> None:
        for name in PLOT_TABS:
            self._set_figure(name, make_empty_figure(name))

    def _set_figure(self, tab_name: str, figure: Figure) -> None:
        old_canvas = self.canvases.pop(tab_name, None)
        old_figure = self.figures.pop(tab_name, None)
        if old_canvas is not None:
            old_canvas.get_tk_widget().destroy()
        if old_figure is not None and old_figure is not figure:
            old_figure.clear()
        frame = self.plot_frames[tab_name]
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.figures[tab_name] = figure
        self.canvases[tab_name] = canvas
