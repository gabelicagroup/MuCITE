"""Run one JSON request without constructing a renderer."""

from pathlib import Path

from src.config import load_config_document
from src.core.simulation import FlowchartPicSimulation


def main() -> None:
    document = load_config_document(Path("configs/headless_smoke.json"))
    simulation = FlowchartPicSimulation(
        document.ion,
        document.simulation,
        particle_backend=document.execution.backend,
    )
    result = simulation.run()
    print(
        f"termination={result.termination_reason} "
        f"survivors={len(result.survivors)} "
        f"shared_grid={simulation.static_grid is simulation.pic_grid}"
    )


if __name__ == "__main__":
    main()
