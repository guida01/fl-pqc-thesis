"""System-level test for the rejection branch in
pytorchexample/server_app.py:SignedFedAvg.aggregate_train.

None of the 5855 rows currently in results/ ever exercise this branch: every
recorded run had `verified == True` for every client, every round. This test
proves the branch actually rejects a tampered client and still lets the
round complete with the remaining honest clients.

It drives the real pytorchexample ServerApp/ClientApp through Flower's
simulation engine (5 supernodes, 2 rounds, ECDSA-256) with client partition 0
tampering its weights after signing via the `tamper-node-index` run_config
option (the one change made to client_app.py alongside this test — see its
docstring/comment there; the option defaults to -1 and is inert unless set).

We use `flwr.simulation.run_simulation._run_simulation` (private API,
imported directly) instead of the public `run_simulation()` wrapper because
the public wrapper always builds an *empty* run_config for both the
ServerApp and ClientApp contexts — it has no way to pass through the
`scheme`/`tamper-node-index`/etc. values our apps require. `_run_simulation`
accepts an explicit `server_app_context` and an `app_dir`, so we fuse
pyproject.toml's [tool.flwr.app.config] with a small override dict via
`get_fused_config_from_dir`, exactly like the `flwr run` CLI does — without
touching the tracked pyproject.toml or results/. This relies on flwr's
internal layout, hence the exact `flwr==1.29.0` pin in pyproject.toml.
"""

import csv
import os
import random
from pathlib import Path

from flwr.app.user_config import UserConfig
from flwr.common import Context, EventType, RecordDict
from flwr.common.config import get_fused_config_from_dir
from flwr.common.typing import Run
from flwr.serverapp.strategy import FedAvg
from flwr.simulation.run_simulation import _run_simulation
from flwr.supercore.constant import NOOP_FEDERATION

from pytorchexample.client_app import app as client_app
from pytorchexample.server_app import app as server_app

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_SUPERNODES = 5
NUM_ROUNDS = 2
TAMPER_NODE_INDEX = 0
SCHEME = "ECDSA-256"


def test_server_rejects_tampered_client(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    override_config = {
        "scheme": SCHEME,
        "num-server-rounds": NUM_ROUNDS,
        "run-number": 1,
        "results-dir": str(results_dir),
        "num-supernodes": NUM_SUPERNODES,
        "tamper-node-index": TAMPER_NODE_INDEX,
    }
    fused_config = get_fused_config_from_dir(Path(APP_DIR), override_config)

    run = Run.create_empty(run_id=random.getrandbits(62))
    run.override_config = override_config
    run.federation = NOOP_FEDERATION

    server_ctx = Context(
        run_id=run.run_id,
        node_id=0,
        node_config=UserConfig(),
        state=RecordDict(),
        run_config=fused_config,
    )

    # Spy on the *base* FedAvg.aggregate_train — the flwr library method that
    # SignedFedAvg.aggregate_train (server_app.py:36, unmodified) calls via
    # `super().aggregate_train(server_round, valid_replies)` — to record how
    # many replies survive rejection filtering each round. Delegates to the
    # real implementation so the simulation's aggregation is unaffected.
    calls: list[tuple[int, int]] = []
    original_aggregate_train = FedAvg.aggregate_train

    def spy_aggregate_train(self, server_round, replies):
        replies = list(replies)
        calls.append((server_round, len(replies)))
        return original_aggregate_train(self, server_round, replies)

    monkeypatch.setattr(FedAvg, "aggregate_train", spy_aggregate_train)

    _run_simulation(
        num_supernodes=NUM_SUPERNODES,
        exit_event=EventType.PYTHON_API_RUN_SIMULATION_LEAVE,
        client_app=client_app,
        server_app=server_app,
        backend_name="ray",
        app_dir=APP_DIR,
        run=run,
        server_app_context=server_ctx,
        is_app=True,
    )

    # --- pipeline did not crash and completed both rounds ---
    assert [server_round for server_round, _ in calls] == list(range(1, NUM_ROUNDS + 1))

    # --- super().aggregate_train() received exactly 4 replies per round ---
    for server_round, n_replies in calls:
        assert n_replies == NUM_SUPERNODES - 1, (
            f"round {server_round}: expected {NUM_SUPERNODES - 1} replies "
            f"forwarded to FedAvg.aggregate_train, got {n_replies}"
        )

    # --- results CSV confirms exactly 1 rejection per round ---
    csv_path = results_dir / f"{SCHEME}.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == NUM_ROUNDS * NUM_SUPERNODES

    by_round: dict[int, list[bool]] = {}
    for row in rows:
        by_round.setdefault(int(row["round"]), []).append(row["verified"] == "True")

    assert set(by_round) == set(range(1, NUM_ROUNDS + 1))
    for server_round, verified_flags in by_round.items():
        assert len(verified_flags) == NUM_SUPERNODES
        n_rejected = verified_flags.count(False)
        assert n_rejected == 1, (
            f"round {server_round}: expected exactly 1 rejection, got {n_rejected}"
        )
        assert verified_flags.count(True) == NUM_SUPERNODES - 1
