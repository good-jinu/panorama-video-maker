"""Run declarative jobs from `jobs/<name>.toml`.

A job file is either a single step (top-level `run = "<step>"` + its parameters) or a pipeline
of `[[steps]]` tables executed in order. Parameters map 1:1 onto the step function's keyword
arguments and are validated before anything runs.

    description = "Tracking shot over background scene"
    run = "composite"
    scenes = ["outputs/background/sprite.png"]
    fg_sheet = "outputs/character/sprite.png"
    output = "outputs/composite/animation_tracking.mp4"

Usage:
    uv run python -m workshop.jobs                      # list jobs
    uv run python -m workshop.jobs job-name             # run a job (name or path to .toml)
    uv run python -m workshop.jobs job-name --dry-run
"""
import argparse
import inspect
import sys
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "jobs"

# Friendly aliases: TOML key -> function parameter
KEY_ALIASES = {
    "composite": {"scenes": "bg_sheets", "char": "fg_sheet"},
}


def _step_registry() -> dict[str, Callable[..., Any]]:
    # Imported lazily so `--list` / validation don't load ONNX Runtime.
    from workshop.clean import clean_sprite_sheet
    from workshop.comparison import build_comparison_2x2
    from workshop.composite import render_composite
    from workshop.imagen import append_scene_to_timeline, generate_scene_bridge
    from workshop.long_composite import render_timeline_job
    from workshop.music_station import build_music_station
    from workshop.recut import recut_sprite_sheet
    from workshop.sprite import generate_sprite_sheet

    return {
        "sprite": generate_sprite_sheet,
        "clean": clean_sprite_sheet,
        "recut": recut_sprite_sheet,
        "composite": render_composite,
        "timeline": render_timeline_job,
        "comparison": build_comparison_2x2,
        "bridge": generate_scene_bridge,
        "add_scene": append_scene_to_timeline,
        "music_station": build_music_station,
    }


def _wants_path(annotation: Any) -> bool:
    """True if a parameter annotation mentions pathlib.Path (Path, Optional[Path], list[Path], ...)."""
    if annotation is Path:
        return True
    return any(_wants_path(a) for a in typing.get_args(annotation))


def _to_path(value: Any) -> Any:
    if isinstance(value, str):
        return ROOT / value if not Path(value).is_absolute() else Path(value)
    if isinstance(value, list):
        return [_to_path(v) for v in value]
    return value


@dataclass
class Step:
    run: str
    params: dict[str, Any] = field(default_factory=dict)

    def bind(self, registry: dict[str, Callable[..., Any]]) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Resolve the step function and validate/coerce params against its signature."""
        if self.run not in registry:
            raise ValueError(f"Unknown step '{self.run}' (expected one of {sorted(registry)})")
        fn = registry[self.run]
        hints = typing.get_type_hints(fn)
        sig = inspect.signature(fn)
        aliases = KEY_ALIASES.get(self.run, {})

        kwargs = {}
        for key, value in self.params.items():
            name = aliases.get(key, key)
            if name not in sig.parameters:
                valid = sorted(set(sig.parameters) | set(aliases))
                raise ValueError(f"[{self.run}] unknown parameter '{key}'. Valid: {', '.join(valid)}")
            kwargs[name] = _to_path(value) if _wants_path(hints.get(name)) else value

        missing = [
            n for n, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty and n not in kwargs
        ]
        if missing:
            raise ValueError(f"[{self.run}] missing required parameter(s): {', '.join(missing)}")
        return fn, kwargs


@dataclass
class Job:
    name: str
    path: Path
    description: str = ""
    steps: list[Step] = field(default_factory=list)

    @classmethod
    def load(cls, name_or_path: str) -> "Job":
        path = Path(name_or_path)
        if path.suffix != ".toml":
            path = JOBS_DIR / f"{name_or_path}.toml"
        elif not path.is_absolute() and not path.exists() and (JOBS_DIR / path.name).exists():
            path = JOBS_DIR / path.name
        if not path.exists():
            raise FileNotFoundError(f"No job '{name_or_path}' ({path}). Available: {', '.join(list_job_names())}")
        data = tomllib.loads(path.read_text())

        description = data.pop("description", "")
        if "steps" in data:
            raw_steps = data.pop("steps")
            if data:
                raise ValueError(f"{path.name}: top-level keys {sorted(data)} not allowed alongside [[steps]]")
        else:
            raw_steps = [data]

        steps = []
        for i, raw in enumerate(raw_steps, 1):
            raw = dict(raw)
            if "run" not in raw:
                raise ValueError(f"{path.name}: step {i} is missing `run = \"<step>\"`")
            steps.append(Step(run=raw.pop("run"), params=raw))
        return cls(name=path.stem, path=path, description=description, steps=steps)

    def execute(self, dry_run: bool = False) -> list[Any]:
        registry = _step_registry()
        bound = [step.bind(registry) for step in self.steps]  # validate everything up front
        results = []
        for i, (fn, kwargs) in enumerate(bound, 1):
            print(f"==> [{self.name}] step {i}/{len(bound)}: {self.steps[i - 1].run}")
            if dry_run:
                for k, v in kwargs.items():
                    print(f"      {k} = {v!r}")
                continue
            results.append(fn(**kwargs))
        return results


def list_job_names() -> list[str]:
    return sorted(p.stem for p in JOBS_DIR.glob("*.toml"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("jobs", nargs="*", help="Job name(s) from jobs/ or path(s) to .toml files")
    ap.add_argument("--dry-run", action="store_true", help="Validate and print resolved parameters only")
    args = ap.parse_args(argv)

    if not args.jobs:
        for name in list_job_names():
            job = Job.load(name)
            steps = " -> ".join(s.run for s in job.steps)
            print(f"  {name:28s} [{steps}] {job.description}")
        return 0

    try:
        jobs = [Job.load(j) for j in args.jobs]
        for job in jobs:
            job.execute(dry_run=args.dry_run)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
