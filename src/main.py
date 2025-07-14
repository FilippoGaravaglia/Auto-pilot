# main.py

from model_loader import BayesianModel
from decision_maker import DecisionMaker

def main():
    # Carica il modello statico
    model = BayesianModel("vehicle_network.xdsl")
    # Inizializza il decision maker (con stato markoviano interno)
    dm = DecisionMaker(model)
    # Simula 5 intervalli temporali
    for t in range(1, 6):
        print(f"=== Step {t} ===")
        dm.step()

if __name__ == "__main__":
    main()
