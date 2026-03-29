import numpy as np
from Src.algorithms.UCB1 import UCB1


# heritage de ucb1 de base, on rajoute la gestion de la volatilite des bras
class VolatileUCB(UCB1):


    def __init__(self, arms=None):

        super().__init__(arms=arms)
        self.name = "VolatileUCB"
        n_arms = len(self.ground_arms)

        # compteurs pour tracker quand chaque bras a ete vu et combien de fois
        self.availability_count = np.zeros(n_arms)
        self.last_seen = np.full(n_arms, -1, dtype=int)
        self.current_iteration = 0

    def run(self, observed_value, user_context=None):

        self.init_choice(observed_value)
        self._update_availability(observed_value)
        self.arm_chosen = self.choose_action()
        return self.arm_chosen

    def _update_availability(self, observation):
        # on met a jour les compteurs de presence pour les bras qu on voit ce tour ci
        for arm in observation["arm_id"]:
            self.availability_count[arm] += 1
            self.last_seen[arm] = self.current_iteration
        self.current_iteration += 1

    def choose_action(self):
        arm_chosen_index = -1

        if np.min(self.arms_payoff_vectors["tries"]) == 0:
            i = 0
            for arm in self.arms_pool["arm_id"]:
                arm_pos = self.ground_arms.index[
                    self.ground_arms["arm_id"] == arm
                ]
                if self.arms_payoff_vectors["tries"][arm_pos] < 1:
                    arm_chosen_index = i
                    break
                i += 1

        if arm_chosen_index == -1:
            arm_pool_size = len(self.arms_pool["arm_id"])
            ucb_values = np.zeros(arm_pool_size) - 1
            total_tries = np.sum(self.arms_payoff_vectors["tries"])

            i = 0
            for arm in self.arms_pool["arm_id"]:
                arm_pos = self.ground_arms.index[
                    self.ground_arms["arm_id"] == arm
                ]
                tries = self.arms_payoff_vectors["tries"][arm_pos]
                avg_reward = (
                    self.arms_payoff_vectors["cumulated_rewards"][arm_pos]
                    / tries
                )

                # bonus d exploration classique ucb1
                ucb1_bonus = np.sqrt(2 * np.log(total_tries) / tries)

                # calcul du bonus d absence inspire de chen 2018
                # plus un bras a ete absent longtemps, plus on veut l explorer
                absence = max(
                    self.current_iteration - self.last_seen[arm] - 1, 0
                )
                avail = max(self.availability_count[arm], 1)
                volatile_bonus = np.sqrt(absence / avail)

                # score final = moyenne + exploration ucb + bonus volatilite
                ucb_values[i] = avg_reward + ucb1_bonus + volatile_bonus
                i += 1

            arm_chosen_index = np.argmax(ucb_values)

        return self.arms_pool["arm_id"][arm_chosen_index]