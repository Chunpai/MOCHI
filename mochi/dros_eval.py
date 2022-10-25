__author__ = "Chunpai W."

# Reward Estimation with DRos

import pickle
import random
from pathlib import Path

import numpy as np
from numpy.random import RandomState

from scipy.stats import spearmanr

from sklearn.linear_model import LogisticRegression
from obp.policy import IPWLearner
from obp.ope import (
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting as IPW,
    DirectMethod as DM,
    DoublyRobust as DR,
)

policy = "random"

train_data = pickle.load(open("../data/MasteryGrids/all_logged_bandit_feedback.pkl", "rb"))
n_rounds = train_data["n_rounds"]
n_actions = train_data['n_actions']
print("train size: {}".format(n_rounds))

# pscore is the propensity score of behavior policy (log-policy)
train_context = train_data["context"]
train_action = train_data["action"]
train_reward = train_data["reward"]
train_pscore = train_data["pscore"]
train_expected_reward = train_data["expected_reward"]
train_action_context = train_data["action_context"].copy()

test_data = pickle.load(open("../data/MasteryGrids/random_bandit_feedback.pkl", "rb"))
n_rounds = test_data["n_rounds"]
print("test size: {}".format(n_rounds))
test_context = test_data["context"]
test_action = test_data["action"]
test_reward = test_data["reward"]

eval_policy = IPWLearner(n_actions=n_actions, base_classifier=LogisticRegression())
eval_policy.fit(
    context=train_context,
    action=train_action,
    reward=train_reward,
    pscore=train_pscore
)

# action dist may be same as action_dist from train_context
action_dist = eval_policy.predict(context=test_context)


regression_model = RegressionModel(
    n_actions=n_actions,
    base_model=LogisticRegression(),
)

estimated_rewards_by_reg_model = regression_model.fit_predict(
    context=test_context,
    action=test_action,
    reward=test_reward,
)

ope = OffPolicyEvaluation(
    bandit_feedback=test_data,
    ope_estimators=[IPW(), DM(), DR()]
)

ope.visualize_off_policy_estimates(
    action_dist=action_dist,
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    fig_dir=Path("."),
    fig_name=policy
)