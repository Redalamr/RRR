"""
Script de test pour vérifier que chaque algorithme fonctionne
sans erreur sur le dataset Mushrooms (02-Mushrooms).

On réduit l'horizon à 500 itérations pour un test rapide.
"""

import sys
import traceback
import pandas as pd

from Src.process.simulator import Simulator


ALGORITHMS_NO_CONTEXT = ["Random", "EGreedy", "UCB1", "TS"]
ALGORITHMS_WITH_CONTEXT = ["LinUCB", "LinTS"]

HORIZON = 500


def get_context_dimension():
    """Détermine la dimension du contexte à partir du dataset Mushrooms."""
    df = pd.read_csv("Resources/bandit_datasets/02-Mushrooms/contexts.csv")
    d = df.shape[1] - 1
    print(f"Context dimension (d) = {d}  (colonnes du fichier contexts.csv moins context_id)")
    return d


def test_algorithm(algo_name, d=None):
    """Instancie le simulateur avec l'algorithme donné et lance une courte simulation."""
    print(f"\n{'='*60}")
    print(f"  TEST : {algo_name}")
    print(f"{'='*60}")

    try:
        if d is not None:
            sim = Simulator(algorithm_name=algo_name, d=d)
        else:
            sim = Simulator(algorithm_name=algo_name)

        sim.horizon = HORIZON
        sim.life_sign_delay = (9999, HORIZON + 1)

        sim.run_simulation()

        accuracy = sim.results.algorithm_performance["accuracy"][HORIZON - 1]
        regrets = sim.results.algorithm_performance["cumulated_regrets"][HORIZON - 1]
        print(f"\n  RESULTAT {algo_name} : accuracy={round(accuracy, 4)}, regrets={round(regrets, 1)}")
        print(f"  >>> OK <<<")
        return True

    except Exception as e:
        print(f"\n  ERREUR avec {algo_name} :")
        traceback.print_exc()
        return False


def main():
    d = get_context_dimension()
    results = {}

    for algo in ALGORITHMS_NO_CONTEXT:
        results[algo] = test_algorithm(algo)

    for algo in ALGORITHMS_WITH_CONTEXT:
        results[algo] = test_algorithm(algo, d=d)

    print(f"\n\n{'='*60}")
    print(f"  RESUME DES TESTS")
    print(f"{'='*60}")
    all_ok = True
    for algo, ok in results.items():
        status = "OK" if ok else "ECHEC"
        print(f"  {algo:10s} : {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\n  Tous les algorithmes fonctionnent correctement !")
    else:
        print(f"\n  Certains algorithmes ont échoué.")
        sys.exit(1)


if __name__ == "__main__":
    main()
