from Src.process.simulator import Simulator

# liste des algos qu on teste dans chaque setting
ALGORITHMS_MAB = ["EGreedy", "Random", "UCB1", "TS"]
ALGORITHMS_CMAB = ["LinUCB", "LinTS"]
ALGORITHMS_VOLATILE = ["VolatileUCB"]
DATASET_NAME = "06-MovieLens"
# dimension du contexte utilisateur sans le context_id
CONTEXT_DIM = 22
NB_RUNS = 10


# on lance les algos classiques avec le masque de volatilite active
def run_volatile():

    for algo_name in ALGORITHMS_MAB:
        print(f"\n{'='*60}")
        print(f"  Algorithme (Volatile - classique) : {algo_name}")
        print(f"{'='*60}")
        sim = Simulator(
            algorithm_name=algo_name,
            dataset_name=DATASET_NAME,
            mode="Volatile"
        )
        sim.run_multiple_simulations(nb_runs=NB_RUNS)

    # pour les cmab on doit passer la dimension du contexte
    for algo_name in ALGORITHMS_CMAB:
        print(f"\n{'='*60}")
        print(f"  Algorithme (Volatile - classique) : {algo_name}")
        print(f"{'='*60}")
        sim = Simulator(
            algorithm_name=algo_name,
            d=CONTEXT_DIM,
            dataset_name=DATASET_NAME,
            mode="Volatile"
        )
        sim.run_multiple_simulations(nb_runs=NB_RUNS)

# on lance notre algo volatile-aware (volatileucb)
def run_volatile_aware():

    for algo_name in ALGORITHMS_VOLATILE:
        print(f"\n{'='*60}")
        print(f"  Algorithme (Volatile-Aware) : {algo_name}")
        print(f"{'='*60}")
        sim = Simulator(
            algorithm_name=algo_name,
            dataset_name=DATASET_NAME,
            mode="Volatile"
        )
        sim.run_multiple_simulations(nb_runs=NB_RUNS)

# baseline sans volatilite, tous les bras dispo a chaque tour
def run_static_baseline():

    for algo_name in ALGORITHMS_MAB:
        print(f"\n{'='*60}")
        print(f"  Algorithme (Baseline statique) : {algo_name}")
        print(f"{'='*60}")
        sim = Simulator(
            algorithm_name=algo_name,
            dataset_name=DATASET_NAME,
            mode="Baseline"
        )
        sim.run_multiple_simulations(nb_runs=NB_RUNS)

    for algo_name in ALGORITHMS_CMAB:
        print(f"\n{'='*60}")
        print(f"  Algorithme (Baseline statique) : {algo_name}")
        print(f"{'='*60}")
        sim = Simulator(
            algorithm_name=algo_name,
            d=CONTEXT_DIM,
            dataset_name=DATASET_NAME,
            mode="Baseline"
        )
        sim.run_multiple_simulations(nb_runs=NB_RUNS)


if __name__ == "__main__":
    # phase 1 : on fait tourner tous les algos en baseline pour avoir une reference
    print("\n" + "#" * 60)
    print("  PHASE 1 : Baseline (tous les bras disponibles)")
    print("#" * 60)
    run_static_baseline()

    # phase 2 : memes algos mais avec le masque de volatilite
    print("\n" + "#" * 60)
    print("  PHASE 2 : Volatile Arms - algorithmes classiques")
    print("#" * 60)
    run_volatile()

    # phase 3 : notre algo adapte a la volatilite
    print("\n" + "#" * 60)
    print("  PHASE 3 : Volatile Arms - algorithmes volatile-aware")
    print("#" * 60)
    run_volatile_aware()