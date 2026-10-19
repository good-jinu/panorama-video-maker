"""Make a panorama journey video from one prompt.

Usage:
    uv run python -m workshop.make "a small car riding through world's cities"
    uv run python -m workshop.make "a red balloon over Europe" --scenes 4 --yes
    uv run python -m workshop.make --project outputs/make/a-small-car-riding   # resume / re-render

Re-running with the same project folder reuses every file already generated. Edit `plan.json`
(or delete a scene/bridge image) and re-run to change only that part.
"""
import argparse
import json
import sys
from pathlib import Path

from workshop.config import settings
from workshop.make.pipeline import MakeOptions, MakeProject
from workshop.make.plan import JourneyPlan, slugify


def _confirm(plan: JourneyPlan, pending: list[str]) -> bool:
    print(f"\nTitle: {plan.title}")
    print(f"Style: {plan.style}")
    print(f"Actor: {plan.actor.name} ({plan.actor.kind.value})")
    print("Scenes: " + " -> ".join(s.name for s in plan.scenes))
    print(f"\nThis run will make {len(pending)} image API call(s):")
    for call in pending:
        print(f"  - {call}")
    if not sys.stdin.isatty():
        print("Not a terminal; pass --yes to proceed.")
        return False
    return input("Proceed? [y/N] ").strip().lower() in ("y", "yes")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m workshop.make", description="Panorama Video Maker: make a journey video from a prompt")
    ap.add_argument("prompt", nargs="?", help="What the video is about")
    ap.add_argument("--project", type=Path, help="Project folder (default: outputs/make/<prompt-slug>)")
    ap.add_argument("--scenes", type=int, default=5, help="Number of scenes to plan (2-8)")
    ap.add_argument("--seconds-per-scene", type=float, default=12.0)
    ap.add_argument("--plan-only", action="store_true", help="Write plan.json and stop")
    ap.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation before image generation")
    args = ap.parse_args()

    if not args.prompt and not args.project:
        ap.error("give a prompt, or --project to resume")
    root = args.project or settings.outputs_dir / "make" / slugify(args.prompt)[:48]

    project = MakeProject(root, MakeOptions(num_scenes=args.scenes, seconds_per_scene=args.seconds_per_scene))
    if args.plan_only:
        plan = project.load_or_plan(args.prompt)
        print(plan.model_dump_json(indent=2))
        print(f"\nSaved {project.plan_path}. {len(project.pending_image_calls(plan))} image call(s) pending.")
        return

    result = project.run(args.prompt, confirm=None if args.yes else _confirm)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
