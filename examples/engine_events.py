"""Observe engine lifecycle events without coupling them to rendering."""

from pathlib import Path

from src.config import load_config_document
from src.core import EventBus
from src.core.simulation import FlowchartPicSimulation


def main() -> None:
    document = load_config_document(Path("configs/headless_smoke.json"))
    events: list[str] = []
    event_bus = EventBus()
    event_bus.subscribe(lambda event: events.append(type(event).__name__))
    simulation = FlowchartPicSimulation(
        document.ion,
        document.simulation,
        particle_backend=document.execution.backend,
        event_bus=event_bus,
    )
    simulation.run()
    print(" -> ".join(events))


if __name__ == "__main__":
    main()
