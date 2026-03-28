import numpy as np


class ResultStorer():
  
    def __init__(self, horizon):

        self.horizon = horizon
        self.start_time = None
        self.end_time = None
        self.simulation_duration = None

        self.threshold = 4
        self.algorithm_performance = {"predicted_arms": np.zeros(horizon),
                                      "correctness": np.zeros(horizon),
                                      "accuracy": np.zeros(horizon),
                                      "cumulated_regrets": np.zeros(horizon)}

        self.all_runs_performance = []

    def reset_current_run(self):

        if np.any(self.algorithm_performance["correctness"]):
            self.all_runs_performance.append({
                "predicted_arms": self.algorithm_performance["predicted_arms"].copy(),
                "correctness": self.algorithm_performance["correctness"].copy(),
                "accuracy": self.algorithm_performance["accuracy"].copy(),
                "cumulated_regrets": self.algorithm_performance["cumulated_regrets"].copy()
            })

        self.algorithm_performance = {"predicted_arms": np.zeros(self.horizon),
                                      "correctness": np.zeros(self.horizon),
                                      "accuracy": np.zeros(self.horizon),
                                      "cumulated_regrets": np.zeros(self.horizon)}
        self.start_time = None
        self.end_time = None
        self.simulation_duration = None

    def store_current_run(self):
        self.all_runs_performance.append({
            "predicted_arms": self.algorithm_performance["predicted_arms"].copy(),
            "correctness": self.algorithm_performance["correctness"].copy(),
            "accuracy": self.algorithm_performance["accuracy"].copy(),
            "cumulated_regrets": self.algorithm_performance["cumulated_regrets"].copy()
        })

    def compute_statistics(self):

        if not self.all_runs_performance:
            return None

        n_runs = len(self.all_runs_performance)

        correctness_matrix = np.zeros((n_runs, self.horizon))
        accuracy_matrix = np.zeros((n_runs, self.horizon))
        regrets_matrix = np.zeros((n_runs, self.horizon))

        for i, run_data in enumerate(self.all_runs_performance):
            correctness_matrix[i] = run_data["correctness"]
            accuracy_matrix[i] = run_data["accuracy"]
            regrets_matrix[i] = run_data["cumulated_regrets"]

        return {
            "mean_correctness": np.mean(correctness_matrix, axis=0),
            "std_correctness": np.std(correctness_matrix, axis=0),
            "mean_accuracy": np.mean(accuracy_matrix, axis=0),
            "std_accuracy": np.std(accuracy_matrix, axis=0),
            "mean_cumulated_regrets": np.mean(regrets_matrix, axis=0),
            "std_cumulated_regrets": np.std(regrets_matrix, axis=0),
            "n_runs": n_runs
        }

    def update_measures(self, iteration, observed_value):

        self.update_correctness(iteration, observed_value)
        self.update_accuracy(iteration)
        self.update_regrets(iteration)

    def update_correctness(self, iteration, observed_value):

        feedback = observed_value["feedback"][observed_value["arm_id"]
                                      == self.algorithm_performance["predicted_arms"][iteration]].iloc[0]

        if feedback >= self.threshold:
            self.algorithm_performance["correctness"][iteration] = 1

    def update_accuracy(self, iteration):
  
        self.algorithm_performance["accuracy"][iteration] = \
                np.sum(self.algorithm_performance["correctness"]) / (iteration + 1)

    def update_regrets(self, iteration):

        if iteration == 0:
            self.algorithm_performance["cumulated_regrets"][iteration] = 1 - self.algorithm_performance["correctness"][0]
        else:
            self.algorithm_performance["cumulated_regrets"][iteration] = \
                self.algorithm_performance["cumulated_regrets"][iteration-1] + (1 - self.algorithm_performance["correctness"][iteration])
