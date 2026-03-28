import numpy as np


class LinUCB():

    def __init__(self, arms=None, d=None, alpha=1.0):

        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "LinUCB"
        self.alpha = alpha
        self.d = d

        self.A = {}
        self.b = {}
        for arm in self.ground_arms["arm_id"]:
            self.A[arm] = np.identity(self.d)
            self.b[arm] = np.zeros(self.d)

        self.arm_chosen = None
        self.last_context = None
        self.threshold = 4

    def run(self, observed_value, user_context=None):

        self.init_choice(observed_value)
        self.last_context = user_context
        self.arm_chosen = self.choose_action(user_context)

        return self.arm_chosen

    def init_choice(self, observation):

        self.arm_chosen = -1
        self.arms_pool = self.ground_arms[self.ground_arms["arm_id"].isin(observation["arm_id"])]
        self.arms_pool.reset_index(inplace=True)

    def choose_action(self, user_context):

        arm_pool_size = len(self.arms_pool['arm_id'])
        ucb_values = np.zeros(arm_pool_size)

        i = 0
        for arm in self.arms_pool['arm_id']:
            A_inv = np.linalg.inv(self.A[arm])
            theta_hat = A_inv @ self.b[arm]
            ucb_values[i] = theta_hat @ user_context + self.alpha * np.sqrt(user_context @ A_inv @ user_context)
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
        x = self.last_context
        self.A[self.arm_chosen] = self.A[self.arm_chosen] + np.outer(x, x)
        self.b[self.arm_chosen] = self.b[self.arm_chosen] + reward_binaire * x
