import time
import warnings

warnings.simplefilter('ignore')

import numpy as np
import pickle
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
import random

# import open bandit pipeline (obp)
import obp
from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function,
    linear_behavior_policy
)
from obp.policy import IPWLearner
from obp.ope import (
    OffPolicyEvaluation,
    RegressionModel,
    DoublyRobustWithShrinkage as DRos,
    DoublyRobustWithShrinkageTuning as DRosTuning,
)

## configurations
num_runs = 100
num_data_list = [100, 200, 400, 800, 1600, 3200]

## define a dataset class
train_data = pickle.load(open("../data/MasteryGrids/train_bandit_feedback.pkl", "rb"))
n_rounds = train_data["n_rounds"]
n_actions = train_data['n_actions']
print("train size: {}".format(n_rounds))

# pscore is the propensity score of behavior policy (log-policy)
train_context = train_data["context"]
train_action = train_data["action"]
train_reward = train_data["reward"]
train_pscore = train_data["pscore"] ### Check this. Try popularity as Log-policy
train_expected_reward = train_data["expected_reward"]
train_action_context = train_data["action_context"].copy()

test_data = pickle.load(open("../data/MasteryGrids/test_bandit_feedback.pkl", "rb"))
n_rounds = test_data["n_rounds"]
print("test size: {}".format(n_rounds))
test_context = test_data["context"]
test_action = test_data["action"]
test_reward = test_data["reward"]
test_pscore = test_data["pscore"]
test_expected_reward = test_data["expected_reward"]
test_action_context = test_data["action_context"].copy()

val_data = pickle.load(open("../data/MasteryGrids/val_bandit_feedback.pkl", "rb"))
n_rounds = val_data["n_rounds"]
print("val size: {}".format(n_rounds))
val_context = val_data["context"]
val_action = val_data["action"]
val_reward = val_data["reward"]
val_pscore = test_data["pscore"]
val_expected_reward = val_data["expected_reward"]
val_action_context = val_data["action_context"].copy()

### evaluation (behavior) policy training
# the goal is to compute the ground-truth policy values on test data
# then use the validation data to compute estimate \hat^{V}(\pi_e)
ipw_learner = IPWLearner(
    n_actions=n_actions,
    base_classifier=LogisticRegression(C=100, max_iter=10000, random_state=12345)
)

### given some bandit feedback data, we could estimate the behavior policy with Logistic Regression
ipw_learner.fit(
    context=train_context,
    action=train_action,
    reward=train_reward,
    # pscore=train_pscore,
)

# predict best action (based on action distribution) on new data (new context)
# shape is (n_rounds, n_actions, 1)
action_dist_ipw_test = ipw_learner.predict(
    context=test_context,
)

# approximated ground-truth policy value based on Monte Carlo
# V(\pi_e): calculate the policy value of given action distribution on the given expected_reward
# scalar value
# expected reward is computed based on:
#   1. under each context, for each action we aggregate number of clicks or non-clicks
#   2. the ratio of click / (click+ nonclick) is the expected reward
policy_value_of_ipw = np.average(test_expected_reward, weights=action_dist_ipw_test[:, :, 0],
                                 axis=1).mean()

## evaluation of OPE estimators
se_df_list = []
for num_data in num_data_list:
    se_list = []
    for _ in tqdm(range(num_runs), desc=f"num_data={num_data}..."):
        ## generate validation data which is the log data
        val_context = val_data["context"][:num_data]
        val_action = val_data["action"][:num_data]
        val_reward = val_data["reward"][:num_data]
        val_pscore = val_data["pscore"][:num_data]
        val_expected_reward = val_data["expected_reward"][:num_data]
        val_action_context = val_data["action_context"].copy()
        validation_bandit_data = {}
        validation_bandit_data["n_rounds"] = num_data
        validation_bandit_data["n_actions"] = n_actions
        validation_bandit_data["position"] = None
        validation_bandit_data["context"] = val_context
        validation_bandit_data["action"] = val_action
        validation_bandit_data["reward"] = val_reward
        validation_bandit_data["pscore"] = val_pscore
        validation_bandit_data["expected_reward"] = val_expected_reward
        validation_bandit_data["action_context"] = val_action_context

        ## make decisions on validation data
        action_dist_ipw_val = ipw_learner.predict(
            context=val_context,
        )

        ## OPE using validation data
        regression_model = RegressionModel(
            n_actions=n_actions,
            base_model=LogisticRegression(C=100, max_iter=10000, random_state=12345)
        )

        ## estimate the \hat{r}
        estimated_rewards = regression_model.fit_predict(
            context=val_context,  # context; x
            action=val_action,  # action; a
            reward=val_reward,  # reward; r
            n_folds=2,  # 2-fold cross fitting
            random_state=12345,
        )

        ### use logged bandit feedback to perform ope
        ope = OffPolicyEvaluation(
            bandit_feedback=validation_bandit_data,
            ## Doubly Robust with optimistic shrinkage (DRos) with built-in hyperparameter tuning.
            # we don't need to care about the hyperparameter tuning procedure
            # but we cannot return the best hyperparameter
            ope_estimators=[
                DRos(lambda_=1, estimator_name="DRos (1)"),
                DRos(lambda_=100, estimator_name="DRos (100)"),
                DRos(lambda_=10000, estimator_name="DRos (10000)"),
                DRosTuning(
                    use_bias_upper_bound=False,
                    lambdas=np.arange(1, 10002, 100).tolist(),
                    estimator_name="DRos (tuning)"
                ),
            ]
        )

        # we want to know how good is \hat^{V}_{DRos} with different lambda
        # compute the \hat^{V}_{DRos} and then compute MSE
        squared_errors = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=policy_value_of_ipw,  # V(\pi_e)
            action_dist=action_dist_ipw_val,  # \pi_e(a|x)
            estimated_rewards_by_reg_model=estimated_rewards,  # \hat{q}(x,a)
            metric="se",  # squared error
        )
        se_list.append(squared_errors)
    ## maximum importance weight in the validation data
    ### a larger value indicates that the logging and evaluation policies are greatly different
    max_iw = (action_dist_ipw_val[
                  np.arange(validation_bandit_data["n_rounds"]),
                  validation_bandit_data["action"],
                  0
              ] / validation_bandit_data["pscore"]).max()
    tqdm.write(f"maximum importance weight={np.round(max_iw, 5)}\n")

    ## summarize results
    se_df = DataFrame(DataFrame(se_list).stack()) \
        .reset_index(1).rename(columns={"level_1": "est", 0: "se"})
    se_df["num_data"] = num_data
    se_df_list.append(se_df)
    tqdm.write("=====" * 15)
    time.sleep(0.5)

# aggregate all results
result_df = pd.concat(se_df_list).reset_index(level=0)

query = "est == 'DRos (1)' or est == 'DRos (100)' or est == 'DRos (10000)'"
xlabels = [100, 800, 1600, 3200]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=5,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df.query(query),
)
# title and legend
ax.legend(
    ["DRos (1)", "DRos (100)", "DRos (10000)"],
    loc="upper right", fontsize=25,
)
# yaxis
ax.set_yscale("log")
ax.set_ylim(3e-2, 0.55)
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.1, 0.5)
# xaxis
# ax.set_xscale("log")
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
plt.show()

plt.clf()
xlabels = [100, 800, 1600, 3200]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.lineplot(
    linewidth=5,
    dashes=False,
    legend=False,
    x="num_data",
    y="se",
    hue="est",
    ax=ax,
    data=result_df,
)
# title and legend
ax.legend(
    ["DRos (1)", "DRos (100)", "DRos (10000)", "DRos (tuning)"],
    loc="upper right", fontsize=22,
)
# yaxis
ax.set_yscale("log")
ax.set_ylim(3e-2, 0.55)
ax.set_ylabel("mean squared error (MSE)", fontsize=25)
ax.tick_params(axis="y", labelsize=15)
ax.yaxis.set_label_coords(-0.1, 0.5)
# xaxis
# ax.set_xscale("log")
ax.set_xlabel("number of samples in the log data", fontsize=25)
ax.set_xticks(xlabels)
ax.set_xticklabels(xlabels, fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.1)
plt.show()
