# decision_maker.py

import random

class DecisionMaker:
    """Implementa la logica decisionale markoviana basata sulla rete Bayesiana."""

    def __init__(self, model):
        """
        :param model: istanza di BayesianModel su slice statica
        """
        self.model = model
        # Stato markoviano iniziale
        self.weather = "dry"       # secco
        self.terrain = "smooth"    # terreno liscio
        self.failure = "ok"        # sensore funzionante
        self.position = "Center"   # veicolo al centro corsia

    def step(self):
        # 1) Transizione di Weather (esempio: P(resta dry)=0.8, P(resta wet)=0.8)
        p_stay_dry = 0.8 if self.weather == "dry" else 0.2
        self.weather = "dry" if random.random() < p_stay_dry else "wet"

        # 2) Transizione di Terrain (esempio: P(resta smooth)=0.7, P(resta rough)=0.7)
        p_stay_smooth = 0.7 if self.terrain == "smooth" else 0.3
        self.terrain = "smooth" if random.random() < p_stay_smooth else "rough"

        # 3) Persistenza + degradazione di SensorFailure
        if self.failure == "ok" and random.random() < 0.1:
            self.failure = "failure"
        # se già 'failure', rimane failure

        # 4) Imposta evidenze sulla rete statica
        self.model.clear_all_evidence()
        self.model.set_evidence("Weather",       self.weather)
        self.model.set_evidence("Terrain",       self.terrain)
        self.model.set_evidence("SensorFailure", self.failure)
        self.model.set_evidence("VehiclePosition", self.position)

        # 5) Inferenza
        self.model.update_beliefs()
        acc_probs      = self.model.get_probabilities("SensorAccuracy")
        reading_probs  = self.model.get_probabilities("SensorReading")
        decision_probs = self.model.get_probabilities("Decision")
        nextpos_probs  = self.model.get_probabilities("VehiclePositionNext")

        # 6) Estrai MAP dalle distribuzioni
        reading  = max(reading_probs,  key=reading_probs.get)
        decision = max(decision_probs, key=decision_probs.get)
        newpos   = max(nextpos_probs,  key=nextpos_probs.get)

        # 7) Stampa risultati del passo
        print(f"Meteo={self.weather}, Terreno={self.terrain}, Failure={self.failure}")
        print(f"  Accuratezza sensore: {acc_probs}")
        print(f"  Lettura sensore: {reading} (dist: {reading_probs})")
        print(f"  Decisione: {decision} (dist: {decision_probs})")
        print(f"  Nuova posizione: {newpos} (dist: {nextpos_probs})\n")

        # 8) Aggiorna stato per il passo successivo
        self.position = newpos
