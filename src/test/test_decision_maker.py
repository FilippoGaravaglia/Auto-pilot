import unittest
from src.model_loader import BayesianModel
from src.decision_maker import DecisionMaker
from src.constants import COSTO_RICERCHE_MERCATO, COSTO_IMPROV_QUALITA


class TestDecisionMaker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Metodo per preparare l'ambiente di test. Carichiamo il modello bayesiano una sola volta.
        """
        cls.model_path = "production_decision_network.xdsl"  # Assicurati che il file sia corretto
        cls.model = BayesianModel(cls.model_path)
        cls.decision_maker = DecisionMaker(cls.model)

    def test_model_loading(self):
        """
        Testa che il modello venga caricato correttamente dall'XDLS.
        """
        self.assertIsNotNone(self.model.network, "Il modello non è stato caricato correttamente")

    def test_probabilities_without_evidence(self):
        """
        Testa le probabilità iniziali (senza evidenze) per il nodo Profit e verifica che siano coerenti con il modello.
        """
        probabilities = self.model.get_node_probabilities("Profit")
        expected_probabilities = [0.1, 0.4, 0.5]  # Aggiorna questi valori in base al tuo modello
        self.assertAlmostEqual(probabilities[0], expected_probabilities[0], delta=0.01)
        self.assertAlmostEqual(probabilities[1], expected_probabilities[1], delta=0.01)
        self.assertAlmostEqual(probabilities[2], expected_probabilities[2], delta=0.01)

    def test_probabilities_with_evidence(self):
        """
        Testa le probabilità per il nodo Profit quando viene impostata un'evidenza (es., qualità alta).
        """
        self.model.set_evidence("Quality", "High")
        probabilities = self.model.get_node_probabilities("Profit")
        self.assertGreater(probabilities[2], 0.5,
                           "La probabilità del profitto alto dovrebbe aumentare con qualità alta.")
        self.model.clear_evidence()  # Ripulisci per test successivi

    def test_valore_atteso_no_actions(self):
        """
        Testa il valore atteso con nessuna azione intrapresa.
        """
        valore_atteso = self.decision_maker.calcola_valore_atteso(miglioramento=False, ricerche=False)
        # Modifica il valore atteso in base al tuo modello
        expected_valore_atteso = (0.4 * 10000) + (0.5 * 50000)  # Aggiorna i numeri!
        self.assertAlmostEqual(valore_atteso, expected_valore_atteso, delta=1)


if __name__ == "__main__":
    unittest.main()
