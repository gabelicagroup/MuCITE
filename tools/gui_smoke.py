"""Construct the real Tk window and dialogs without showing them on screen."""

from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.render.gui.main_window import MainWindow
from src.render.gui.beam_dialog import BeamSetupDialog
from src.render.gui.field_dialog import FieldBakerDialog
from src.render.gui.iict_dialog import IictSettingsDialog
from src.render.gui.runtime_dialogs import PicSettingsDialog, CollisionSettingsDialog


def main() -> None:
    real_tk, real_top = tk.Tk, tk.Toplevel

    def hidden_root(*args, **kwargs):
        root = real_tk(*args, **kwargs)
        root.withdraw()
        return root

    def hidden_top(*args, **kwargs):
        top = real_top(*args, **kwargs)
        top.withdraw()
        return top

    with patch.object(tk, "Tk", hidden_root), patch.object(tk, "Toplevel", hidden_top):
        window = MainWindow()
        try:
            window.root.update_idletasks()
            config = window.config
            constructors = [
                lambda: BeamSetupDialog(window.root, config.beam, lambda value: None),
                lambda: FieldBakerDialog(window.root, config.field_bake, config.runtime, lambda *args: None),
                lambda: PicSettingsDialog(window.root, config.runtime, lambda value: None),
                lambda: CollisionSettingsDialog(window.root, config.runtime, lambda value: None),
                lambda: IictSettingsDialog(window.root, config.runtime, lambda value: None),
            ]
            for construct in constructors:
                dialog = construct()
                window.root.update_idletasks()
                print(type(dialog).__name__ + " OK")
                dialog.top.destroy()
        finally:
            window.root.destroy()
    print("GUI_CONSTRUCTION_SMOKE_OK")


if __name__ == "__main__":
    main()
