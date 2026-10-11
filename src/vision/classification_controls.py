"""Predeclared drift controls for formal CPU experiments (stdlib only).

Only complete, contemporaneously bracketed model/block groups can be resumed.
Rejected measurements remain in an archive and never enter ranked aggregates.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import time


class ControlError(ValueError):
    pass


def drift_factor(before, after):
    if before.get("status") != "passed" or after.get("status") != "passed":
        return None
    first = [sample["inference_ms"] for sample in before["samples"]]
    last = [sample["inference_ms"] for sample in after["samples"]]
    if not first or not last or len(first) != len(last):
        return None
    left, right = statistics.median(first), statistics.median(last)
    if left <= 0 or right <= 0:
        return None
    return max(left, right) / min(left, right)


def control_job(run, directory, model_id, block, role, attempt, images, budget, make_job):
    baseline = {"id": f"{model_id}.B0.t1", "model_id": model_id,
                "variant_id": "B0", "threads": 1}
    job = make_job(run, directory, baseline, block, "control", images, budget)
    job["job_name"] = f"{model_id}.B0.t1.control.{role}.a{attempt}.b{block}"
    job.update(control_role=role, attempt=attempt,
               output_path=str(directory / "controls" / f"{job['job_name']}.json"))
    return job


def _nominal_job(run, directory, experiment, block, attempt, images, budget, make_job):
    job = make_job(run, directory, experiment, block, "inference", images, budget)
    job["attempt"] = attempt
    return job


def archive_job(job, directory, archive):
    """Move only this generated task's files; preserve original bytes."""
    archive.mkdir(parents=True, exist_ok=True)
    output = Path(job["output_path"])
    for path in (output, output.with_suffix(".samples.jsonl"),
                 directory / "jobs" / f"{job['job_name']}.json"):
        if path.exists():
            suffix = ".job.json" if path.parent.name == "jobs" else path.suffix
            name = f"{path.stem}{suffix}"
            target = archive / name
            if target.exists():
                raise ControlError("Refusing to overwrite an archived task")
            path.replace(target)


def run_formal_groups(run, directory, experiments, images, budget, *, make_job,
                      run_job, resume_result, write_json, resuming=False,
                      threshold=1.10, max_attempts=2):
    directory = Path(directory)
    controls = directory / "controls"
    controls.mkdir(exist_ok=True)
    ledger_path = controls / "ledger.json"
    identity = {name: run[name] for name in ("config_sha256", "manifest_sha256", "code_fingerprint")}
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if any(ledger.get(key) != value for key, value in identity.items()):
            raise ControlError("Control ledger provenance changed")
        if ledger.get("threshold") != threshold or ledger.get("max_attempts") != max_attempts:
            raise ControlError("Control protocol changed")
    else:
        ledger = {"schema_version": 1, **identity, "threshold": threshold,
                  "max_attempts": max_attempts, "groups": {}}
        write_json(ledger_path, ledger)
    results = []
    model_ids = sorted({entry["model_id"] for entry in experiments})
    timeout = run["config"]["worker_timeout_s"]
    for block in range(budget["repeats"]):
        model_order = list(model_ids)
        random.Random(run["config"]["seed"] + block).shuffle(model_order)
        for model_id in model_order:
            group_key = f"{model_id}.b{block}"
            members = [entry for entry in experiments if entry["model_id"] == model_id]
            random.Random(run["config"]["seed"] + block).shuffle(members)
            prior = ledger["groups"].get(group_key)
            if resuming and prior and prior["status"] == "accepted":
                restored = []
                for experiment in members:
                    job = _nominal_job(run, directory, experiment, block, prior["attempt"],
                                       images, budget, make_job)
                    record = resume_result(job, directory)
                    if record is None or not record.get("environment_accepted"):
                        raise ControlError("Accepted model/block is missing its measured result")
                    restored.append(record)
                results.extend(restored)
                continue
            if resuming and prior and prior["status"] == "unstable":
                for experiment in members:
                    job = _nominal_job(run, directory, experiment, block, prior["attempt"], images, budget, make_job)
                    results.append(json.loads(Path(job["output_path"]).read_text(encoding="utf-8")))
                continue
            # An interrupted model/block cannot reuse measurements bracketed by
            # controls from another invocation. Archive it and start anew.
            if resuming:
                archive = directory / "rejected-results" / f"{group_key}.interrupted-{time.time_ns()}"
                for attempt in range(max_attempts):
                    for experiment in members:
                        archive_job(_nominal_job(run, directory, experiment, block, attempt,
                                                 images, budget, make_job), directory, archive)
                    for role in ("before", "after"):
                        archive_job(control_job(run, directory, model_id, block, role, attempt,
                                                images, budget, make_job), directory, archive)
            for attempt in range(max_attempts):
                before_job = control_job(run, directory, model_id, block, "before", attempt, images, budget, make_job)
                before = run_job(before_job, directory, timeout)
                group_results = []
                for experiment in members:
                    job = _nominal_job(run, directory, experiment, block, attempt, images, budget, make_job)
                    record = run_job(job, directory, timeout)
                    group_results.append((job, record))
                    print(f"formal block {block + 1}/{budget['repeats']} a{attempt} {experiment['id']}: {record['status']}", flush=True)
                after_job = control_job(run, directory, model_id, block, "after", attempt, images, budget, make_job)
                after = run_job(after_job, directory, timeout)
                factor = drift_factor(before, after)
                accepted = factor is not None and factor <= threshold
                decision = {"attempt": attempt, "status": "accepted" if accepted else "unstable",
                            "drift_factor": factor, "threshold": threshold,
                            "before_path": before_job["output_path"], "after_path": after_job["output_path"],
                            "before_pid": before.get("worker_pid"), "after_pid": after.get("worker_pid"),
                            "before_identity": before.get("worker_identity"),
                            "after_identity": after.get("worker_identity")}
                if accepted:
                    for job, record in group_results:
                        record.update(environment_accepted=True, attempt=attempt,
                                      drift_factor=factor, drift_control_group=group_key)
                        write_json(Path(job["output_path"]), record, overwrite=True)
                        results.append(record)
                    ledger["groups"][group_key] = decision
                    write_json(ledger_path, ledger, overwrite=True)
                    break
                rejected = directory / "rejected-results" / f"{group_key}.a{attempt}-{time.time_ns()}"
                for job, record in group_results:
                    archive_job(job, directory, rejected)
                if attempt + 1 < max_attempts:
                    print(f"{group_key}: predeclared drift gate failed ({factor}); retrying group once", flush=True)
                    continue
                for job, record in group_results:
                    failed = {**record, "status": "failed_runtime", "samples": [], "sample_count": 0,
                              "environment_accepted": False, "drift_factor": factor,
                              "error": "Predeclared before/after control gate failed after bounded retry",
                              "rejected_result_directory": str(rejected)}
                    failed.pop("samples_path", None)
                    failed.pop("samples_sha256", None)
                    write_json(Path(job["output_path"]), failed)
                    write_json(directory / "jobs" / f"{job['job_name']}.json", job)
                    results.append(failed)
                ledger["groups"][group_key] = decision
                write_json(ledger_path, ledger, overwrite=True)
    return results
