#!/usr/bin/env python3
"""Safely resume an interrupted final campaign."""

import argparse
import csv
from pathlib import Path
import subprocess


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, choices=(5, 10), required=True)
    return p.parse_args()


def runner_for(clients):
    if clients == 5:
        import run_all_schemes as runner
    else:
        import run_10clients as runner
    return runner


def paths(runner, scheme):
    name = scheme.replace("/", "_")
    return (
        runner.RESULTS_DIR / f"{name}.csv",
        runner.RESULTS_DIR / f"{name}_eval.csv",
    )


def load_csv(path):
    if not path.exists():
        return None, []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames

        if not fields or "run" not in fields:
            raise RuntimeError(f"Missing 'run' column: {path}")

        rows = list(reader)

    for line_no, row in enumerate(rows, start=2):
        raw = (row.get("run") or "").strip()

        try:
            int(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid run value {raw!r} in {path}:{line_no}"
            ) from exc

    return fields, rows


def count_run(path, run_num):
    _, rows = load_csv(path)
    return sum(int(row["run"]) == run_num for row in rows)


def status(runner, scheme, run_num):
    update_csv, eval_csv = paths(runner, scheme)

    updates = count_run(update_csv, run_num)
    evals = count_run(eval_csv, run_num)

    expected_updates = (
        runner.NUM_ROUNDS * runner.NUM_SUPERNODES
    )
    expected_evals = runner.NUM_ROUNDS + 1

    if updates > expected_updates or evals > expected_evals:
        raise RuntimeError(
            f"{scheme} run {run_num} exceeds expected counts: "
            f"updates={updates}/{expected_updates}, "
            f"eval={evals}/{expected_evals}. Refusing recovery."
        )

    if (
        updates == expected_updates
        and evals == expected_evals
    ):
        return "complete", updates, evals

    if updates == 0 and evals == 0:
        return "missing", updates, evals

    return "partial", updates, evals


def drop_run(path, run_num):
    fields, rows = load_csv(path)

    if fields is None:
        return

    kept = [
        row
        for row in rows
        if int(row["run"]) != run_num
    ]

    temp = path.with_name(path.name + ".resume_tmp")

    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(kept)

    temp.replace(path)


def clean_partial(runner, scheme, run_num):
    update_csv, eval_csv = paths(runner, scheme)

    drop_run(update_csv, run_num)
    drop_run(eval_csv, run_num)


def reject_future_rows(runner, scheme, run_num):
    for future in range(
        run_num + 1,
        runner.N_RUNS + 1,
    ):
        _, updates, evals = status(
            runner,
            scheme,
            future,
        )

        if updates or evals:
            raise RuntimeError(
                f"Found future rows for {scheme} run {future}; "
                "automatic recovery is refusing this unusual state."
            )


def require_clean_pyproject(runner):
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "--",
            "pyproject.toml",
        ],
        cwd=runner.PROJECT_DIR,
    )

    if result.returncode == 1:
        raise SystemExit(
            "ERROR: pyproject.toml is modified.\n"
            "Run: git restore pyproject.toml"
        )

    if result.returncode != 0:
        raise SystemExit(
            "ERROR: could not check pyproject.toml with git."
        )


def main():
    opts = args()
    runner = runner_for(opts.clients)

    if (
        not runner.RESULTS_DIR.exists()
        or not any(runner.RESULTS_DIR.iterdir())
    ):
        raise SystemExit(
            "ERROR: resume requires an existing non-empty directory:\n"
            f"  {runner.RESULTS_DIR}"
        )

    require_clean_pyproject(runner)

    runner.check_environment()
    runner.check_cpu_governor()

    original = runner.PYPROJECT.read_text()

    total = (
        len(runner.SCHEMES)
        * runner.N_RUNS
    )

    position = 0

    runner.log_execution(
        f"SAFE RESUME requested ({opts.clients} clients)"
    )

    try:
        for run_num in range(
            1,
            runner.N_RUNS + 1,
        ):
            order = runner.randomized_scheme_order(
                run_num
            )

            for scheme in order:
                position += 1

                state, updates, evals = status(
                    runner,
                    scheme,
                    run_num,
                )

                if state == "complete":
                    runner.log_execution(
                        f"[{position}/{total}] "
                        f"skip {scheme} "
                        f"run {run_num}: complete"
                    )
                    continue

                reject_future_rows(
                    runner,
                    scheme,
                    run_num,
                )

                if state == "partial":
                    runner.log_execution(
                        f"[{position}/{total}] "
                        f"clean partial {scheme} "
                        f"run {run_num}: "
                        f"updates={updates}, eval={evals}"
                    )

                    clean_partial(
                        runner,
                        scheme,
                        run_num,
                    )

                    if (
                        status(
                            runner,
                            scheme,
                            run_num,
                        )[0]
                        != "missing"
                    ):
                        raise RuntimeError(
                            f"Cleanup failed for "
                            f"{scheme} run {run_num}."
                        )

                runner.log_execution(
                    f"[{position}/{total}] "
                    f"resume {scheme} "
                    f"run {run_num}"
                )

                if not runner.run_scheme(
                    scheme,
                    run_num,
                ):
                    raise RuntimeError(
                        f"Resume stopped at "
                        f"{scheme} run {run_num}. "
                        "Stop remaining Flower processes, "
                        "restore pyproject.toml, then "
                        "run this command again."
                    )

        runner.print_final_summary()

    finally:
        runner.PYPROJECT.write_text(
            original
        )

        subprocess.run(
            [
                "ray",
                "stop",
                "--force",
            ],
            capture_output=True,
            cwd=runner.PROJECT_DIR,
        )


if __name__ == "__main__":
    main()
