from pathlib import Path

import pytest

from workshop import jobs
from workshop.jobs import ROOT, Job, list_job_names


def write(tmp_path, text, name="job.toml") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_single_step_job_with_aliases_and_paths(tmp_path):
    job = Job.load(str(write(tmp_path, '''
description = "demo"
run = "composite"
scenes = ["outputs/a.png", "/abs/b.png"]
char = "outputs/cat.png"
output = "outputs/out.mp4"
scale = 0.5
''')))
    assert job.description == "demo" and len(job.steps) == 1
    fn, kwargs = job.steps[0].bind(jobs._step_registry())
    assert fn.__name__ == "render_composite"
    assert kwargs["bg_sheets"] == [ROOT / "outputs/a.png", Path("/abs/b.png")]
    assert kwargs["fg_sheet"] == ROOT / "outputs/cat.png"
    assert kwargs["scale"] == 0.5


def test_pipeline_job(tmp_path):
    job = Job.load(str(write(tmp_path, '''
[[steps]]
run = "recut"
sheet_path = "a.png"
output_path = "b.png"
frames_range = "1-2"
columns = 2

[[steps]]
run = "clean"
input_path = "b.png"
''')))
    assert [s.run for s in job.steps] == ["recut", "clean"]


@pytest.mark.parametrize("body, message", [
    ('run = "composite"\nscenes = ["a"]\nchar = "b"\noutput = "c"\nbogus = 1', "unknown parameter 'bogus'"),
    ('run = "recut"\nsheet_path = "a.png"', "missing required"),
    ('run = "nope"', "Unknown step"),
])
def test_validation_errors(tmp_path, body, message):
    job = Job.load(str(write(tmp_path, body)))
    with pytest.raises(ValueError, match=message):
        job.execute(dry_run=True)


def test_missing_run_key(tmp_path):
    with pytest.raises(ValueError, match="missing `run"):
        Job.load(str(write(tmp_path, 'scale = 1')))


@pytest.mark.parametrize("name", list_job_names())
def test_repo_jobs_are_valid(name):
    Job.load(name).execute(dry_run=True)
