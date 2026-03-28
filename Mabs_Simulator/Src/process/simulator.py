'''
Created on 23 mars 2024

@author: aletard
'''

import time
import numpy as np
import pandas as pd
import random as rd

from Src.utils.repository_manager import RepositoryManager as RM
from Src.data_management.data_loader import DataLoader as DL
from Src.Reporting.report_generator import ReportGenerator
from Src.Reporting.results_storer import ResultStorer

from Src.algorithms.EGreedy import EGreedy
from Src.algorithms.Random import Random
from Src.algorithms.UCB1 import UCB1
from Src.algorithms.TS import TS
from Src.algorithms.LinUCB import LinUCB
from Src.algorithms.LinTS import LinTS


class Simulator():

    ALGORITHM_MAP = {
        "EGreedy": EGreedy,
        "Random": Random,
        "UCB1": UCB1,
        "TS": TS,
        "LinUCB": LinUCB,
        "LinTS": LinTS,
    }

    def __init__(self, algorithm_name="EGreedy", d=None, dataset_name="02-Mushrooms"):
        print("Initializing Simulator")

        self.dataset_name = dataset_name
        self.datas = self.data_extraction()

        if algorithm_name not in self.ALGORITHM_MAP:
            raise ValueError(f"Algorithme inconnu : {algorithm_name}")

        algorithm_class = self.ALGORITHM_MAP[algorithm_name]

        if algorithm_name in ("LinUCB", "LinTS"):
            if d is None:
                raise ValueError(f"Le paramètre d est obligatoire pour {algorithm_name}")
            self.algorithm = algorithm_class(self.datas["arms"], d=d)
        else:
            self.algorithm = algorithm_class(self.datas["arms"])

        self.horizon = 30000
        self.results = ResultStorer(self.horizon)
        self.reporter = ReportGenerator(
            RM.create_repository_with_timestamp("../Output"),
            (self.dataset_name, self.horizon, self.algorithm.name)
        )

        self.life_sign_delay = (300, 5000)
        self._arm_rng = np.random.default_rng(seed=42)
        self._base_probs = None
        self._cycle_periods = None

    def run_simulation(self):
        print("starting simulation")

        self.results.start_time = time.time()
        for iteration in range(self.horizon):

            user_id = rd.choice(self.datas["contexts"]["context_id"])
            user_context = self.context_formatter(
                self.datas["contexts"][self.datas["contexts"]['context_id'] == user_id]
            )
            observed_value = self.datas["results"][
                self.datas["results"]["context_id"] == user_id
            ].copy()

            mask = self.generate_availability_mask(self.datas["arms"]["arm_id"], iteration)
            active_arm_ids = self.datas["arms"]["arm_id"][mask == 1].values
            observed_value = observed_value[observed_value["arm_id"].isin(active_arm_ids)].copy()

            user_context = user_context[1:]
            self.results.algorithm_performance["predicted_arms"][iteration] = self.algorithm.run(observed_value, user_context)
            self.algorithm.update(observed_value)
            self.results.update_measures(iteration, observed_value)

            if (time.time() - self.results.start_time == self.life_sign_delay[0]) | \
                    (iteration % self.life_sign_delay[1] == 0):
                self.sign_life(iteration)

        self.results.end_time = time.time()
        self.end_sign()

    def run_multiple_simulations(self, nb_runs=10):
        algorithm_name = None
        d_param = None
        for name, cls in self.ALGORITHM_MAP.items():
            if isinstance(self.algorithm, cls):
                algorithm_name = name
                break

        if algorithm_name in ("LinUCB", "LinTS"):
            d_param = self.algorithm.d

        all_regret_curves = np.zeros((nb_runs, self.horizon))

        for run_index in range(nb_runs):
            print(f"\n=== Run {run_index + 1} / {nb_runs} ===")

            if d_param is not None:
                self.algorithm = self.ALGORITHM_MAP[algorithm_name](self.datas["arms"], d=d_param)
            else:
                self.algorithm = self.ALGORITHM_MAP[algorithm_name](self.datas["arms"])

            self.results = ResultStorer(self.horizon)
            self._arm_rng = np.random.default_rng(seed=42 + run_index)
            self._base_probs = None
            self._cycle_periods = None

            self.run_simulation()

            all_regret_curves[run_index] = self.results.algorithm_performance["cumulated_regrets"]

        mean_regret = np.mean(all_regret_curves, axis=0)
        std_regret = np.std(all_regret_curves, axis=0)

        print(f"\n=== Résultats sur {nb_runs} runs ===")
        print(f"Regrets cumulés par run : {all_regret_curves[:, -1]}")
        print(f"Moyenne du regret cumulé final : {round(float(mean_regret[-1]), 3)}")
        print(f"Écart-type du regret cumulé final : {round(float(std_regret[-1]), 3)}")

        statistics = {
            "mean_cumulated_regrets": mean_regret,
            "std_cumulated_regrets": std_regret,
            "n_runs": nb_runs
        }
        self.reporter.plot_averaged_regret(statistics, algorithm_name)

        return {
            "all_regret_curves": all_regret_curves,
            "mean_regret": mean_regret[-1],
            "std_regret": std_regret[-1],
        }

    def data_extraction(self):
        rss_path = RM.get_absolute_from_relative_path(f"../Resources/bandit_datasets/{self.dataset_name}")
        files_to_load = RM.get_files_in_directory(rss_path)
        files_path = []
        for file in files_to_load:
            files_path.append(f"{rss_path}/{file}")

        return DL.load_multiple_files(files_path)

    def context_formatter(self, context):
        try:
            context = context.drop(["context_id"], axis=1)
        except:
            print("Error on context formatting")

        nb_dimensions = context.shape[1]
        context = np.array(context)
        user_context = context.reshape(nb_dimensions,)

        return user_context

    def generate_availability_mask(self, arms, iteration):
        n_arms = len(arms)

        if self._base_probs is None or len(self._base_probs) != n_arms:
            rng = np.random.default_rng(seed=123)
            self._base_probs = rng.uniform(0.5, 0.95, size=n_arms)
            self._cycle_periods = rng.integers(50, 500, size=n_arms)

        phase = 2 * np.pi * iteration / self._cycle_periods
        effective_probs = np.clip(self._base_probs + 0.2 * np.sin(phase), 0.05, 1.0)

        mask = (self._arm_rng.random(n_arms) < effective_probs).astype(int)

        if mask.sum() == 0:
            mask[self._arm_rng.integers(n_arms)] = 1

        return mask

    def sign_life(self, iteration):
        sign_life_message = (
            f"\nSimulator has been running for {round(time.time() - self.results.start_time, 3)} seconds. \n"
            f"Currently going for iteration {iteration}, "
            f"latest accuracy value : {round(self.results.algorithm_performance['accuracy'][iteration], 3)}, "
            f"cumulated regrets: {round(self.results.algorithm_performance['cumulated_regrets'][iteration], 3)}.\n\n"
        )
        self.reporter.log_generator(sign_life_message)

    def end_sign(self):
        end_message = (
            f"\nSimulation correctly ended. \n"
            f"The simulation has been running for {round(self.results.end_time - self.results.start_time, 3)} seconds. \n"
            f"The simulation included {self.horizon} iterations, "
            f"latest accuracy value : {round(self.results.algorithm_performance['accuracy'][self.horizon - 1], 3)}, "
            f"cumulated regrets: {round(self.results.algorithm_performance['cumulated_regrets'][self.horizon - 1], 3)}.\n\n"
        )
        self.reporter.log_generator(end_message)