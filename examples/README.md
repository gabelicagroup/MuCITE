# Examples

Run from the repository root after installing the dependencies:

```powershell
python -m examples.headless_json
python -m examples.engine_events
```

Both use the explicit independent-backend configuration in
`configs/headless_smoke.json`. The first demonstrates the headless engine API;
the second subscribes to lifecycle events. Production studies need your own
validated field and ion parameters.
