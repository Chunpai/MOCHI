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
    SelfNormalizedInverseProbabilityWeighting as SNIPW,
    DirectMethod as DM,
    DoublyRobust as DR,
    DoublyRobustWithShrinkage as DRos
)

# policy = "random"
# policy = "instruct_seq"
# policy = "inv_instruct_seq"
# policy = "mastery"
policy = "highest_prob_correct"

data = pickle.load(open("../data/MasteryGrids/data.pkl", "rb"))
pretest_posttest_dict = data["pretest_posttest_dict"]

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

test_data = pickle.load(open("../data/MasteryGrids/{}_bandit_feedback.pkl".format(policy), "rb"))
n_rounds = test_data["n_rounds"]
print("test size: {}".format(n_rounds))
test_context = test_data["context"]
test_action = test_data["action"]
test_reward = test_data["reward"]
user_index_dict = test_data["user_index_dict"]

eval_policy = IPWLearner(n_actions=n_actions,
                         base_classifier=LogisticRegression(tol=100, max_iter=1000))
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
    base_model=LogisticRegression(tol=1e-4, max_iter=100),
)

estimated_rewards_by_reg_model = regression_model.fit_predict(
    context=test_context,
    action=test_action,
    reward=test_reward,
)

print(estimated_rewards_by_reg_model.shape)

ope = OffPolicyEvaluation(
    bandit_feedback=test_data,
    ope_estimators=[IPW(), SNIPW(), DM(), DR(), DRos(lambda_=10.)]
    # ope_estimators = [IPW(), DM(), DR()]
)

estimated_round_reward_df = ope.visualize_off_policy_estimates(
    action_dist=action_dist,
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    fig_dir=Path("."),
    fig_name=policy
)

estimated_round_reward = estimated_round_reward_df.to_numpy()
pickle.dump(estimated_round_reward, open("{}_estimated_round_reward.pkl".format(policy), "wb"))

estimated_round_reward = pickle.load(open("{}_estimated_round_reward.pkl".format(policy), "rb"))
score_growth_list = []
posttest_list = []

avg_ipw_list = []
avg_snipw_list = []
avg_dm_list = []
avg_dr_list = []
avg_dros_list = []
for user in pretest_posttest_dict:
    pretest, posttest = pretest_posttest_dict[user]
    posttest_list.append(posttest)
    score_growth_list.append(posttest-pretest)
    index_list = user_index_dict[user]
    for col in range(5):
        user_est_reward = estimated_round_reward[index_list, col]
        avg_user_est_reward = np.mean(user_est_reward)
        if col == 0:
            avg_ipw_list.append(avg_user_est_reward)
        elif col == 1:
            avg_snipw_list.append(avg_user_est_reward)
        elif col == 2:
            avg_dm_list.append(avg_user_est_reward)
        elif col == 3:
            avg_dr_list.append(avg_user_est_reward)
        elif col == 4:
            avg_dros_list.append(avg_user_est_reward)
        else:
            raise ValueError


for col in range(5):
    if col == 0:
        print("ipw")
        correlation_post_score = spearmanr(avg_ipw_list, posttest_list)
        correlation_score_growth = spearmanr(avg_ipw_list, score_growth_list)
        print("corr post score: {}".format(correlation_post_score))
        print("corr score growth: {}".format(correlation_score_growth))
    elif col == 1:
        print("snipw")
        correlation_post_score = spearmanr(avg_snipw_list, posttest_list)
        correlation_score_growth = spearmanr(avg_snipw_list, score_growth_list)
        print("corr post score: {}".format(correlation_post_score))
        print("corr score growth: {}".format(correlation_score_growth))
    elif col == 2:
        print("dm")
        correlation_post_score = spearmanr(avg_dm_list, posttest_list)
        correlation_score_growth = spearmanr(avg_dm_list, score_growth_list)
        print("corr post score: {}".format(correlation_post_score))
        print("corr score growth: {}".format(correlation_score_growth))
    elif col == 3:
        print("dr")
        correlation_post_score = spearmanr(avg_dr_list, posttest_list)
        correlation_score_growth = spearmanr(avg_dr_list, score_growth_list)
        print("corr post score: {}".format(correlation_post_score))
        print("corr score growth: {}".format(correlation_score_growth))
    elif col == 4:
        print("dros")
        correlation_post_score = spearmanr(avg_dros_list, posttest_list)
        correlation_score_growth = spearmanr(avg_dros_list, score_growth_list)
        print("corr post score: {}".format(correlation_post_score))
        print("corr score growth: {}".format(correlation_score_growth))
    else:
        raise ValueError

