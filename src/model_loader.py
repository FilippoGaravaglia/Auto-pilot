# model_loader.py

import pysmile
import pysmile_license  # assicurati di avere la licenza valida

class BayesianModel:
    """Classe per gestire l'interazione con la rete Bayesiana."""

    def __init__(self, network_path: str):
        self.network = pysmile.Network()
        try:
            self.network.read_file(network_path)
            self.network.update_beliefs()
        except pysmile.SMILEException as e:
            raise RuntimeError(f"Errore caricamento {network_path}: {e}")

    def clear_all_evidence(self):
        self.network.clear_all_evidence()

    def set_evidence(self, variable: str, value: str):
        self.network.set_evidence(variable, value)

    def update_beliefs(self):
        self.network.update_beliefs()

    def get_probabilities(self, node: str) -> dict:
        outcomes = self.network.get_outcome_ids(node)
        probs = self.network.get_node_value(node)
        return dict(zip(outcomes, probs))
