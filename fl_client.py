import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import io
from signature_manager import SignatureManager

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_weights(model):
    return [p.detach().numpy() for p in model.parameters()]

def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data = torch.tensor(w)

def weights_to_bytes(weights):
    buf = io.BytesIO()
    np.save(buf, np.array(weights, dtype=object), allow_pickle=True)
    return buf.getvalue()


class SignedFlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id, train_loader, scheme):
        self.client_id = client_id
        self.train_loader = train_loader
        self.model = SimpleCNN()
        self.sig_mgr = SignatureManager(scheme)
        self.scheme = scheme
        print(f"[Client {client_id}] Keygen done for {scheme} in {self.sig_mgr.keygen_time:.4f}s")

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for images, labels in self.train_loader:
            optimizer.zero_grad()
            loss = criterion(self.model(images), labels)
            loss.backward()
            optimizer.step()

        updated_weights = get_weights(self.model)
        payload = weights_to_bytes(updated_weights)
        signature, sign_time = self.sig_mgr.sign(payload)

        print(f"[Client {self.client_id}] Signed in {sign_time:.4f}s | sig size: {len(signature)} bytes")

        return updated_weights, len(self.train_loader.dataset), {
            "client_id":   str(self.client_id),
            "scheme":      self.scheme,
            "signature":   signature.hex(),
            "public_key":  self.sig_mgr.public_key_bytes.hex(),
            "sign_time":   sign_time,
            "sig_size":    len(signature),
            "pubkey_size": len(self.sig_mgr.public_key_bytes),
        }

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        return 0.0, len(self.train_loader.dataset), {"accuracy": 0.0}
