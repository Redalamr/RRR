from Src.process.simulator import Simulator

ALGORITHMS_MAB = ["EGreedy", "Random", "UCB1","TS"]
ALGORITHMS_CMAB = ["LinUCB", "LinTS"]
DATASET_NAME = "06-MovieLens"
CONTEXT_DIM = 22
NB_RUNS = 10


def run_static_baseline():
    for algo_name in ALGORITHMS_MAB:
        print(f"\n{'='*60}")
        print(f"  Algorithme (Baseline statique) : {algo_name}")
        print(f"{'='*60}")
        sim = Simulator(algorithm_name=algo_name, dataset_name=DATASET_NAME)
        sim.use_volatile_mask = False
        sim.run_multiple_simulations(nb_runs=NB_RUNS)

    for algo_name in ALGORITHMS_CMAB:
        print(f"\n{'='*60}")
        print(f"  Algorithme (Baseline statique) : {algo_name}")
        print(f"{'='*60}")
        sim = Simulator(algorithm_name=algo_name, d=CONTEXT_DIM, dataset_name=DATASET_NAME)
        sim.use_volatile_mask = False
        sim.run_multiple_simulations(nb_runs=NB_RUNS)


if __name__ == "__main__":
    run_static_baseline()