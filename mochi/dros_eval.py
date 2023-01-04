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
import matplotlib.pyplot as plt

# policy = "random"
# policy = "instruct_seq"
# policy = "mastery"
# policy = "highest_prob_correct"
# policy = "myopic"
policy = "pear"

data = pickle.load(open("../data/MasteryGrids/data.pkl", "rb"))
pretest_posttest_dict = data["pretest_posttest_dict"]

data = pickle.load(open("../data/MasteryGrids/logged_bandit_feedback_{}_action_dist.pkl".format(
    policy), "rb"))
n_rounds = data["n_rounds"]
n_actions = data['n_actions']
print("train size: {}".format(n_rounds))

# pscore is the propensity score of behavior policy (log-policy)
logged_belief_context = data["belief_context"]
logged_attempt_context = data["attempt_context"]
logged_correct_context = data["correct_context"]

logged_action = data["action"]
logged_reward = data["reward"]
logged_pscore = data["pscore"]
logged_action_context = data["action_context"].copy()
eval_policy_action_dist = data["eval_policy_action_dist"]
user_index_dict = data["user_index_dict"]

max_iw = (eval_policy_action_dist[np.arange(n_rounds), logged_action, 0] / logged_pscore).max()
print(f"maximum importance weight={np.round(max_iw, 5)}\n")

regression_model = RegressionModel(
    n_actions=n_actions,
    base_model=LogisticRegression(tol=0.0001, max_iter=1000, solver="liblinear")
    # base_model = LogisticRegression(tol=0.0001, max_iter=100, solver="newton-cg")
)

context = np.concatenate([logged_belief_context, logged_attempt_context, logged_correct_context],
                         axis=1)
# context = np.concatenate([logged_attempt_context, logged_correct_context],
#                          axis=1)

data["context"] = context  # remember to change this


ope = OffPolicyEvaluation(
    bandit_feedback=data,
    ope_estimators=[IPW(), SNIPW(), DM(), DR(), DRos(lambda_=1.)]
    # ope_estimators = [IPW(), DM(), DR()]
)

# K-Fold cross-validation included in fit_predict function
estimated_rewards_by_reg_model = regression_model.fit_predict(
    context=context,  # should we estimate the context based on BKT?
    action=logged_action,
    reward=logged_reward,
    n_folds=2
)
# this is reward data frame for all possible actions at each context
print(estimated_rewards_by_reg_model.shape)
print(np.isnan(estimated_rewards_by_reg_model).any())

# estimated_policy_value = ope.estimate_policy_values(
#     action_dist=action_dist,
#     estimated_rewards_by_reg_model=estimated_rewards_by_reg_model)
# print(estimated_policy_value)




all_reward_dict = {}
for fold in range(1, 6):
    rewards_dict = pickle.load(open("fold_{}_rewards.pkl".format(fold), "rb"))
    for user in rewards_dict:
        if user not in all_reward_dict:
            all_reward_dict[user] = rewards_dict[user]
user_list = list(user_index_dict.keys())
estimated_rewards_by_lstm = all_reward_dict[user_list[0]]
for user in user_list[1:]:
    est_reward = all_reward_dict[user]
    estimated_rewards_by_lstm = np.concatenate((estimated_rewards_by_lstm, est_reward), axis=0)
estimated_rewards_by_lstm = np.expand_dims(estimated_rewards_by_lstm, axis=-1)




estimated_round_reward_df = ope.visualize_off_policy_estimates(
    action_dist=eval_policy_action_dist,
    # estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    estimated_rewards_by_reg_model=estimated_rewards_by_lstm,
    fig_dir=Path("outputs/"),
    fig_name=policy
)
estimated_round_reward = estimated_round_reward_df.to_numpy()

pickle.dump(estimated_round_reward,
            open("outputs/{}_estimated_round_reward.pkl".format(policy), "wb"))

estimated_round_reward = pickle.load(
    open("outputs/{}_estimated_round_reward.pkl".format(policy), "rb"))
score_growth_list = []
posttest_list = []

print(np.isnan(estimated_round_reward).any())

# plt.clf()
# user_length_list = []
# for user in user_index_dict:
#     user_length_list.append(len(user_index_dict[user]))
# plt.hist(user_length_list)
# plt.show()


### User Length Correlation
avg_ipw_list = []
avg_snipw_list = []
avg_dm_list = []
avg_dr_list = []
avg_dros_list = []
for user in pretest_posttest_dict:
    if len(user_index_dict[user]) > 120:
        continue
    # elif len(user_index_dict[user]) > 100:
    #     continue
    pretest, posttest = pretest_posttest_dict[user]
    posttest_list.append(posttest)
    score_growth_list.append(posttest - pretest)
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

print("number of users: {}, {}".format(len(posttest_list), len(score_growth_list)))
for col in range(5):
    if col == 0:
        print("ipw")
        correlation_post_score = spearmanr(avg_ipw_list, posttest_list)
        correlation_score_growth = spearmanr(avg_ipw_list, score_growth_list)
        print("mean: {:.5f} std: {:.5f}, {}".format(np.mean(avg_ipw_list), np.std(avg_ipw_list),
                                                    [np.round(v, 3) for v in avg_ipw_list]))
        print(" ,{}".format(posttest_list))
        print(" ,{}".format(score_growth_list))
        corr = correlation_post_score[0]
        pv = correlation_post_score[1]
        print("corr: {:.5f}, pv: {:.5f}".format(corr, pv))
        corr = correlation_score_growth[0]
        pv = correlation_score_growth[1]
        print("corr: {:.5f}, pv: {:.5f}".format(corr, pv))
    elif col == 1:
        print("snipw")
        correlation_post_score = spearmanr(avg_snipw_list, posttest_list)
        correlation_score_growth = spearmanr(avg_snipw_list, score_growth_list)
        print("mean: {:.5f} std: {:.5f}, {}".format(np.mean(avg_snipw_list), np.std(avg_snipw_list),
                                                    [np.round(v, 3) for v in avg_snipw_list]))
        print(" ,{}".format(posttest_list))
        print(" ,{}".format(score_growth_list))
        corr = correlation_post_score[0]
        pv = correlation_post_score[1]
        print("corr: {:.5f}, pv: {:.5f}".format(corr, pv))
        corr = correlation_score_growth[0]
        pv = correlation_score_growth[1]
        print("corr: {:.5f}, pv: {:.5f}".format(corr, pv))
    elif col == 2:
        print("dm")
        correlation_post_score = spearmanr(avg_dm_list, posttest_list)
        correlation_score_growth = spearmanr(avg_dm_list, score_growth_list)
        print("mean: {:.5f} std: {:.5f}, {}".format(np.mean(avg_dm_list), np.std(avg_dm_list),
                                                    [np.round(v, 3) for v in avg_dm_list]))
        print(" ,{}".format(posttest_list))
        print(" ,{}".format(score_growth_list))
        corr = correlation_post_score[0]
        pv = correlation_post_score[1]
        print("corr: {:.5f}, pv: {:.5f}".format(corr, pv))
        corr = correlation_score_growth[0]
        pv = correlation_score_growth[1]
        print("corr: {:.5f}, pv: {:.5f}".format(corr, pv))
    elif col == 3:
        print("dr")
        correlation_post_score = spearmanr(avg_dr_list, posttest_list)
        correlation_score_growth = spearmanr(avg_dr_list, score_growth_list)
        print("mean: {:.5f} std: {:.5f}, {}".format(np.mean(avg_dr_list), np.std(avg_dr_list),
                                                    [np.round(v, 3) for v in avg_dr_list]))
        print(" ,{}".format(posttest_list))
        print(" ,{}".format(score_growth_list))
        corr = correlation_post_score[0]
        pv = correlation_post_score[1]
        print("corr: {:.5f}, pv: {:.5f}".format(corr, pv))
        corr = correlation_score_growth[0]
        pv = correlation_score_growth[1]
        print("corr: {:.5f}, pv: {:.5f}".format(corr, pv))
    elif col == 4:
        print("dros")
        correlation_post_score = spearmanr(avg_dros_list, posttest_list)
        correlation_score_growth = spearmanr(avg_dros_list, score_growth_list)
        print("mean: {:.5f} std: {:.5f}, {}".format(np.mean(avg_dros_list), np.std(avg_dros_list),
                                                    [np.round(v, 3) for v in avg_dros_list]))
        print(" ,{}".format(posttest_list))
        print(" ,{}".format(score_growth_list))
        corr = correlation_post_score[0]
        pv = correlation_post_score[1]
        print("corr: {:.5f}, pv: {:.5f}".format(corr, pv))
        corr = correlation_score_growth[0]
        pv = correlation_score_growth[1]
        print("corr: {:.5f}, pv: {:.5f}".format(corr, pv))
    else:
        raise ValueError

# Learning Length Correlation
# test_index = 100
# avg_ipw_list = []
# avg_snipw_list = []
# avg_dm_list = []
# avg_dr_list = []
# avg_dros_list = []
# for user in pretest_posttest_dict:
#     pretest, posttest = pretest_posttest_dict[user]
#     user_length = len(user_index_dict[user])
#     if user_length > test_index:
#         ratio = float(test_index) / user_length
#         posttest = pretest + (posttest-pretest) * ratio
#     posttest_list.append(posttest)
#     score_growth_list.append(posttest-pretest)
#     index_list = user_index_dict[user]
#     for col in range(5):
#         user_est_reward = estimated_round_reward[index_list, col]
#         avg_user_est_reward = np.mean(user_est_reward)
#         if col == 0:
#             avg_ipw_list.append(avg_user_est_reward)
#         elif col == 1:
#             avg_snipw_list.append(avg_user_est_reward)
#         elif col == 2:
#             avg_dm_list.append(avg_user_est_reward)
#         elif col == 3:
#             avg_dr_list.append(avg_user_est_reward)
#         elif col == 4:
#             avg_dros_list.append(avg_user_est_reward)
#         else:
#             raise ValueError
#
#
# print("number of users: {}, {}".format(len(posttest_list), len(score_growth_list)))
# for col in range(5):
#     if col == 0:
#         print("ipw")
#         correlation_post_score = spearmanr(avg_ipw_list, posttest_list)
#         correlation_score_growth = spearmanr(avg_ipw_list, score_growth_list)
#         print("corr post score: {}".format(correlation_post_score))
#         print("corr score growth: {}".format(correlation_score_growth))
#     elif col == 1:
#         print("snipw")
#         correlation_post_score = spearmanr(avg_snipw_list, posttest_list)
#         correlation_score_growth = spearmanr(avg_snipw_list, score_growth_list)
#         print("corr post score: {}".format(correlation_post_score))
#         print("corr score growth: {}".format(correlation_score_growth))
#     elif col == 2:
#         print("dm")
#         correlation_post_score = spearmanr(avg_dm_list, posttest_list)
#         correlation_score_growth = spearmanr(avg_dm_list, score_growth_list)
#         print("corr post score: {}".format(correlation_post_score))
#         print("corr score growth: {}".format(correlation_score_growth))
#     elif col == 3:
#         print("dr")
#         correlation_post_score = spearmanr(avg_dr_list, posttest_list)
#         correlation_score_growth = spearmanr(avg_dr_list, score_growth_list)
#         print("corr post score: {}".format(correlation_post_score))
#         print("corr score growth: {}".format(correlation_score_growth))
#     elif col == 4:
#         print("dros")
#         correlation_post_score = spearmanr(avg_dros_list, posttest_list)
#         correlation_score_growth = spearmanr(avg_dros_list, score_growth_list)
#         print("corr post score: {}".format(correlation_post_score))
#         print("corr score growth: {}".format(correlation_score_growth))
#     else:
#         raise ValueError
