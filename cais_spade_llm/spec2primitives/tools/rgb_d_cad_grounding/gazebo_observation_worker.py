"""Run one Gazebo observation capture in a sourced ROS2 environment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cais_spade_llm.spec2primitives.tools.observation_context import (
    ObservationContextError,
)
from cais_spade_llm.spec2primitives.tools.rgb_d_cad_grounding.gazebo_observation_provider import (
    GazeboObservationProviderError,
    capture_gazebo_observation,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Capture one observation and return its result through a private JSON file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("observations_root", type=Path)
    parser.add_argument("observation_ref")
    parser.add_argument("timeout_sec", type=float)
    parser.add_argument("result_path", type=Path)
    arguments = parser.parse_args(argv)

    try:
        captured_path = capture_gazebo_observation(
            arguments.observations_root,
            arguments.observation_ref,
            timeout_sec=arguments.timeout_sec,
        )
    except GazeboObservationProviderError as exc:
        result = {
            "status": "failed",
            "error_type": "provider",
            "reason": exc.reason,
            "message": str(exc),
        }
    except ObservationContextError as exc:
        result = {
            "status": "failed",
            "error_type": "observation_context",
            "message": str(exc),
        }
    else:
        result = {
            "status": "captured",
            "captured_path": str(captured_path),
        }

    arguments.result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
