import random
import numpy as np


class UCB1():

    def __init__(self, arms=None):

        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "UCB1"

        self.arms_payoff_vectors = {"cumulated_rewards" : np.zeros(len(self.ground_arms)),
                                    "tries" : np.zeros(len(self.ground_arms))
                                    }

        self.arm_chosen = None
        self.threshold = 4

    def run(self, observed_value, user_context=None):

        self.init_choice(observed_value)
        self.arm_chosen = self.choose_action()

        return self.arm_chosen

    def init_choice(self, observation):

        self.arm_chosen = -1
        self.arms_pool = self.ground_arms[self.ground_arms["arm_id"].isin(observation["arm_id"])]
        self.arms_pool.reset_index(inplace=True)

    def choose_action(self):

        arm_chosen_index = -1

        if np.min(self.arms_payoff_vectors["tries"]) == 0:
            i = 0
            for arm in self.arms_pool['arm_id']:
                arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm]

                if self.arms_payoff_vectors["tries"][arm_pos] < 1:
                    arm_chosen_index = i
                    break
                i += 1

        if arm_chosen_index == -1:
            arm_pool_size = len(self.arms_pool['arm_id'])
            ucb_values = np.zeros(arm_pool_size) - 1
            total_tries = np.sum(self.arms_payoff_vectors["tries"])
            i = 0
            for arm in self.arms_pool['arm_id']:
                arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm]
                avg_reward = self.arms_payoff_vectors["cumulated_rewards"][arm_pos] / \
                                self.arms_payoff_vectors["tries"][arm_pos]
                exploration_bonus = np.sqrt(2 * np.log(total_tries) / self.arms_payoff_vectors["tries"][arm_pos])
                ucb_values[i] = avg_reward + exploration_bonus
                i += 1
            arm_chosen_index = np.argmax(ucb_values)

        arm_chosen = self.arms_pool["arm_id"][arm_chosen_index]

        return arm_chosen

    def evaluate(self, observation):

        reward = 0
        feedback = observation["feedback"][observation["arm_id"] == self.arm_chosen].iloc[0]
        if feedback >= self.threshold:
            reward = 1

        return reward

    def update(self, observation):

        feedback = observation["feedback"][observation["arm_id"] == self.arm_chosen].iloc[0]
        reward_binaire = 1 if feedback >= self.threshold else 0
        self.arms_payoff_vectors["cumulated_rewards"][self.arm_chosen] += reward_binaire
        self.arms_payoff_vectors["tries"][self.arm_chosen] += 1
