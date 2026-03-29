import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Src.utils.repository_manager import RepositoryManager as RM


class ReportGenerator():

    def __init__(self, output_repository, simulator_config):

        self.output_repositiory_path = output_repository

        RM.create_repository(f"{self.output_repositiory_path}/logs")
        self.logs_path = RM.get_absolute_from_relative_path(
            f"{self.output_repositiory_path}/logs/logs.txt"
        )

        RM.create_repository(f"{self.output_repositiory_path}/results")
        self.results_path = RM.get_absolute_from_relative_path(
            f"{self.output_repositiory_path}/results"
        )

        RM.create_repository(f"{self.output_repositiory_path}/config")
        self.config_report(
            RM.get_absolute_from_relative_path(
                f"{self.output_repositiory_path}/config/config.txt"
            ),
            simulator_config
        )

    def log_generator(self, message):
        print(message)
        try:
            with open(self.logs_path, "a", encoding="utf-8") as logs:
                sys.stdout = logs
                print(message)
                sys.stdout = sys.__stdout__
        except Exception:
            with open(self.logs_path, "w", encoding="utf-8") as logs:
                sys.stdout = logs
                print(message)
                sys.stdout = sys.__stdout__

    def config_report(self, config_path, simulator_config):
        # simulator_config is a tuple of 3 or 4 elements:
        # (dataset, iterations, algorithm) or
        # (dataset, iterations, algorithm, mode)
        dataset, iterations, algorithm = simulator_config[:3]
        mode = simulator_config[3] if len(simulator_config) > 3 else "Unknown"

        message = (
            f"Simulation configuration: \n"
            f"Dataset: {dataset}, {iterations} iterations, "
            f"algorithm: {algorithm}\n"
            f"Mode: {mode}"
        )
        print(message)
        try:
            with open(config_path, "a", encoding="utf-8") as config:
                sys.stdout = config
                print(message)
                sys.stdout = sys.__stdout__
        except Exception:
            with open(config_path, "w", encoding="utf-8") as config:
                sys.stdout = config
                print(message)
                sys.stdout = sys.__stdout__

    def plot_averaged_regret(self, statistics, algorithm_name):
        mean_regret = statistics["mean_cumulated_regrets"]
        std_regret = statistics["std_cumulated_regrets"]
        n_runs = statistics["n_runs"]
        iterations = np.arange(len(mean_regret))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            iterations, mean_regret, linewidth=2,
            label=f"{algorithm_name} (moyenne)"
        )
        ax.fill_between(
            iterations,
            mean_regret - std_regret,
            mean_regret + std_regret,
            alpha=0.25,
            label="± 1 écart-type"
        )
        ax.set_xlabel("Itérations", fontsize=12)
        ax.set_ylabel("Regret cumulé", fontsize=12)
        ax.set_title(
            f"Regret cumulé moyen — {algorithm_name} ({n_runs} runs)",
            fontsize=14
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        output_path = (
            f"{self.results_path}/averaged_regret_{algorithm_name}.png"
        )
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Graphique sauvegardé : {output_path}")