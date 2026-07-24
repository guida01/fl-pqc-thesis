"""fl_pqc: A Flower / PyTorch app."""

import time
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import CIFAR10CNN, DEVICE, load_data, train, weights_to_bytes
from pytorchexample.signature_manager import SignatureManager


app = ClientApp()


@app.train()
def train_fn(msg: Message, context: Context):
    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size     = context.run_config["batch-size"]
    local_epochs   = context.run_config["local-epochs"]
    scheme         = context.run_config["scheme"]
    run_number     = int(context.run_config.get("run-number", 1))

    model = CIFAR10CNN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(DEVICE)

    trainloader = load_data(partition_id, num_partitions, batch_size, run_number)

    t0 = time.perf_counter()
    train_loss = train(model, trainloader, local_epochs, context.run_config["learning-rate"], DEVICE)
    train_time = time.perf_counter() - t0

    server_round = int(msg.content["config"]["server-round"])
    node_id      = msg.metadata.dst_node_id

    if scheme == "no_signature":
        signature    = b""
        public_key   = b""
        keygen_time  = 0.0
        sign_time    = 0.0
        sig_size     = 0
        pubkey_size  = 0
        payload_size = 0.0
    else:
        sig_mgr     = SignatureManager(scheme)
        keygen_time = sig_mgr.keygen_time          # ← FIX 6: usar timer interno

        payload = (
            weights_to_bytes(model.state_dict())
            + server_round.to_bytes(4, "big")
            + str(node_id).encode("utf-8")
        )
        signature, sign_time = sig_mgr.sign(payload)
        public_key   = sig_mgr.public_key_bytes
        sig_size     = len(signature)
        pubkey_size  = len(public_key)
        payload_size = float(len(payload))

    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "signature": ConfigRecord({
            "signature":  signature.hex(),
            "public_key": public_key.hex(),
            "scheme":     scheme,
        }),
        "metrics": MetricRecord({
            "keygen_time":  keygen_time,
            "sign_time":    sign_time,
            "train_time":   train_time,
            "train_loss":   train_loss,
            "payload_size": payload_size,
            "sig_size":     float(sig_size),
            "pubkey_size":  float(pubkey_size),
            "num-examples": float(len(trainloader.dataset)),
        }),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context):
    content = RecordDict({"metrics": MetricRecord({"eval_loss": 0.0, "num-examples": 0.0})})
    return Message(content=content, reply_to=msg)
