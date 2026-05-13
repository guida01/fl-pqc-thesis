"""fl_pqc: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import SimpleCNN, DEVICE, load_data, train, weights_to_bytes
from pytorchexample.signature_manager import SignatureManager


app = ClientApp()


@app.train()
def train_fn(msg: Message, context: Context):
    """Train the model, sign the weights, return signed update."""
    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size     = context.run_config["batch-size"]
    local_epochs   = context.run_config["local-epochs"]
    scheme         = context.run_config["scheme"]

    # load model from server message
    model = SimpleCNN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(DEVICE)

    # load data and train
    trainloader = load_data(partition_id, num_partitions, batch_size)
    train(model, trainloader, local_epochs, context.run_config["learning-rate"], DEVICE)

    # sign the updated weights
    sig_mgr          = SignatureManager(scheme)
    payload          = weights_to_bytes(model.state_dict())
    signature, sign_time = sig_mgr.sign(payload)

    # pack weights + signature into the reply message
    arrays         = ArrayRecord(model.state_dict())
    signature_info = ConfigRecord({
        "signature":  signature.hex(),
        "public_key": sig_mgr.public_key_bytes.hex(),
        "scheme":     scheme,
    })
    metrics = MetricRecord({
        "sign_time":   sign_time,
        "sig_size":    float(len(signature)),
        "pubkey_size": float(len(sig_mgr.public_key_bytes)),
        "num_examples": float(len(trainloader.dataset)),
    })

    content = RecordDict({
        "arrays":    arrays,
        "signature": signature_info,
        "metrics":   metrics,
    })
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate_fn(msg: Message, context: Context):
    """Stub — evaluation not used in this thesis."""
    content = RecordDict({"metrics": MetricRecord({"eval_loss": 0.0})})
    return Message(content=content, reply_to=msg)