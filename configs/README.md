# Configuration files

- `default.json`: explicit public CLI demo; no external field or IonSPA required.
- `headless_smoke.json`: the same small request under a stable validation name.
- `iict_lite_parameters.example.json`: editable demonstration heat-capacity,
  pseudo-atom mass, and fragmentation models. Not a fitted ion parameter set.

Simulation documents use schema version 1; GUI project JSON uses its own schema
version 5 and must be loaded through the GUI project/config loader. They are not
interchangeable. Paths in runtime requests follow the existing project-root
convention. Use absolute paths for external datasets. CSV paths inside an IICT
parameter document are relative to that parameter document.

The public demo is intentionally different from the original unpublished
default experiment. Legacy configurations must explicitly select a backend;
requesting a removed bundled provider raises an error. The package contains a
copy of the default/demo parameters for wheel installations. Keep those files
in sync when updating the release defaults.
