import numpy as np


class TS():

    def __init__(self, arms=None):

        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "TS"

        self.successes = np.zeros(len(self.ground_arms))
        self.failures = np.zeros(len(self.ground_arms))

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

        arm_pool_size = len(self.arms_pool['arm_id'])
        theta_samples = np.zeros(arm_pool_size)

        i = 0
        for arm in self.arms_pool['arm_id']:
            arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm]
            alpha = self.successes[arm_pos] + 1
            beta = self.failures[arm_pos] + 1
            theta_samples[i] = np.random.beta(alpha, beta)
            i += 1

        arm_chosen_index = np.argmax(theta_samples)
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
        if reward_binaire == 1:
            self.successes[self.arm_chosen] += 1
        else:
            self.failures[self.arm_chosen] += 1
