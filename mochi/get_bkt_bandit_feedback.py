__author__ = "Chunpai W."
__email__ = "cwang25@albany.edu"

# Policies are built based on BKT/DKM plan model, which is used to trace student's knowledge based on
# historical performance.
# We generate the bandit feedback based on the BKT belief as context or historical correctness
# on questions as context.


import argparse
import pickle
import numpy as np
from numpy.random import RandomState

from mochi.bkt import getYudelsonModel, compute_bkt_last_belief
from mochi.bkt import BKTModel
from mochi.rewards import LogisticModel, LinearModel, RidgeModel, SVRModel, LassoModel
from mochi.utils import *
from scipy.stats import spearmanr

from deepkt.utils.config import *
from deepkt.agents import *
from scipy import spatial


def get_data(data_str):
    data = pickle.load(open("../data/{}/data.pkl".format(data_str), "rb"))
    user_list = data["users"]
    item_list = data['items']
    user_records_dict = data["records"]
    pretest_posttest_dict = data["pretest_posttest_dict"]
    next_item_distr_dict = data["next_items_dict"]
    return user_list, item_list, user_records_dict, pretest_posttest_dict, next_item_distr_dict


def generate_problem_seq(user_records):
    first_questions = {}
    next_q_dist = {}
    user_list = list(user_records.keys())
    for user in user_list:
        questions = user_records[user]
        first_q = questions[0]
        if first_q not in first_questions:
            first_questions[first_q] = 0
        first_questions[first_q] += 1
        for i in range(len(questions) - 1):
            curr_q = questions[i]
            next_q = questions[i + 1]
            if curr_q not in next_q_dist:
                next_q_dist[curr_q] = {}
            if next_q not in next_q_dist[curr_q]:
                next_q_dist[curr_q][next_q] = 0
            next_q_dist[curr_q][next_q] += 1
    print(len(next_q_dist))
    for curr_q in next_q_dist:
        sum_count = sum(next_q_dist[curr_q].values())
        for next_q in next_q_dist[curr_q]:
            count = next_q_dist[curr_q][next_q]
            next_q_dist[curr_q][next_q] = float(count) / sum_count
    return next_q_dist


def generate_instructional_sequence(problem_topic_mapping):
    problem_seq = []
    topic_problem_mapping = {}
    for problem in problem_topic_mapping:
        topic = problem_topic_mapping[problem]
        if topic not in topic_problem_mapping:
            topic_problem_mapping[topic] = []
        topic_problem_mapping[topic].append(problem)

    topic_seq = ["Variables",
                 "Primitive_Data_Types",
                 "Constants",
                 "Arithmetic_Operations",
                 "Strings",
                 "Boolean_Expressions",
                 "Decisions",
                 "Switch",
                 "Exceptions",
                 "Loops_while",
                 "Loops_do_while",
                 "Loops_for",
                 "Nested_Loops",
                 "Objects",
                 "Classes",
                 "Arrays",
                 "Two-dimensional_Arrays",
                 "ArrayList",
                 "Inheritance",
                 "Interfaces",
                 "Wrapper_Classes"
                 ]
    for topic in topic_seq:
        problem_seq += sorted(topic_problem_mapping[topic])
    return problem_seq


def eval_policy(bkt_model, policy, traj_lengths, pretest_list, users_question_list,
                users_answer_list, users_name_list, pear_model=None, Q_matrix=None):
    users, items, user_records, pretest_posttest_dict, next_item_distr_dict = get_data(
        "MasteryGrids")
    next_q_dist = generate_problem_seq(user_records)

    data = {}
    n_actions = len(problem_seq)
    data["n_actions"] = n_actions

    prBlue("Policy: {}, Number of Test Users: {}, Number of Actions: {}".format(
        policy, len(users_name_list), n_actions))

    user_index_dict = {}
    belief_context = []
    attempt_context = []
    correct_context = []
    action = []
    reward = []
    pscore = []
    eval_policy_action_dist_list = []

    for traj, user_name in enumerate(users_name_list):
        # reset the test student models' initial belief when new trajectory is sampled as the
        # training user's initial belief
        if user_name not in user_index_dict:
            user_index_dict[user_name] = []
        pretest = pretest_list[traj]
        user_q_list = users_question_list[traj]
        user_a_list = users_answer_list[traj]
        bkt_model.resetState(pretest)  # for plan model, we don't reset the pretest score

        q_list = []
        a_list = []
        curr_attempt_context = [0.] * n_actions
        curr_correct_context = [0.] * n_actions
        if policy == "pear":
            pear_q_list = [problem_name_problem_id_mapping[user_q_list[0]]+1]
            pear_a_list = [user_a_list[0]]
        for t in range(traj_lengths[traj] if type(traj_lengths) == list else traj_lengths):
            # generate an action that follows policy (for BKT-MP, it is based on curr_bel)
            # once an action is generated, it is removed from the action space (problem_list)
            curr_belief = list(np.round(bkt_model.curr_bel[:, 1], 4))

            if t + 1 < len(user_q_list):
                current_q = user_q_list[t]
                current_a = user_a_list[t]
                qid = problem_name_problem_id_mapping[current_q]
                curr_attempt_context[qid] += 1.
                if current_a == 1.:
                    curr_correct_context[qid] = 1.

                if policy == "random":
                    action_dist = [[1. / n_actions] for _ in range(n_actions)]
                elif policy == "instruct_seq":
                    action_dist = [[0.] for _ in range(n_actions)]
                    index = problem_seq.index(current_q)
                    if index + 1 != len(problem_seq):
                        action_dist[index + 1][0] = 1.
                    else:
                        action_dist[0][0] = 1.
                elif policy == "inv_instruct_seq":
                    action_dist = [[0.] for _ in range(n_actions)]
                    index = problem_seq.index(current_q)
                    if index - 1 != -1:
                        action_dist[index - 1][0] = 1.
                    else:
                        action_dist[-1][0] = 1.
                elif policy == "pear":
                    knowledge_state = pear_model.get_knowledge_state(pear_q_list, pear_a_list)
                    mastery = knowledge_state[-1, :].tolist()
                    question_info = []
                    batch_pred_q_list = []
                    batch_pred_a_list = []
                    for index, question_correlation in enumerate(Q_matrix):
                        qid = index + 1
                        pred_q_list = q_list + [qid]
                        pred_a_list = a_list + [1]
                        batch_pred_q_list.append(pred_q_list)
                        batch_pred_a_list.append(pred_a_list)
                    batch_pred_outputs = pear_model.batch_predict(batch_pred_q_list,
                                                                  batch_pred_a_list)
                    proximity_list = []
                    for index, pred_output in enumerate(batch_pred_outputs):
                        pred_q_list = batch_pred_q_list[index]
                        # qid = index + 1
                        question_correlation = Q_matrix[index]
                        if len(pred_q_list) >= pear_model.data_loader.max_seq_len:
                            corr_belief = pred_output[-1].item()
                        else:
                            corr_belief = pred_output[len(pred_q_list)].item()
                        gap = spatial.distance.cosine(mastery, question_correlation)
                        proximity = (1. - gap) * (1. - corr_belief)  # higher is better
                        proximity_list.append(proximity)
                        # question_info.append([qid, gap, corr_belief, proximity])
                        # sorted_question_proximity = sorted(question_info, key=lambda x: x[-1],
                        #                                    reverse=True)
                    min_prox = np.min(proximity_list)
                    proximity_list = np.array(proximity_list) - min_prox
                    total_prox = np.sum(proximity_list)
                    proximity_list = proximity_list / total_prox
                    action_dist = [[0.] for _ in range(n_actions)]
                    assert len(proximity_list) == len(action_dist)
                    for ind in range(len(action_dist)):
                        action_dist[ind][0] = proximity_list[ind]
                else:
                    action_dist = bkt_model.plan(policy)
                    if np.isnan(action_dist).any():
                        print("here")

                user_index_dict[user_name].append(len(belief_context))
                belief_context.append(curr_belief.copy())
                attempt_context.append(curr_attempt_context.copy())
                correct_context.append(curr_correct_context.copy())
                eval_policy_action_dist_list.append(action_dist)
                true_next_question = problem_name_problem_id_mapping[user_q_list[t + 1]]
                if len(next_q_dist[current_q]) == 0:
                    a_i = problem_name_problem_id_mapping[problem_seq[0]]
                    propensity = 1.
                else:
                    sorted_items = sorted(next_q_dist[current_q].items(), key=lambda x: x[1],
                                          reverse=True)
                    a_i = problem_name_problem_id_mapping[sorted_items[0][0]]
                    propensity = sorted_items[0][1]
                action.append(a_i)
                pscore.append(propensity)
                if true_next_question == a_i:
                    r = 1
                else:
                    r = 0
                reward.append(r)

            true_q = user_q_list[t]
            true_a = user_a_list[t]
            topic_name = problem_topic_mapping[true_q]
            step_outcomes = [(true_q, true_a, 0)]
            if policy == "pear":
                pear_q_list.append(problem_name_problem_id_mapping[true_q]+1)
                pear_a_list.append(true_a)
            bkt_model.updateState(step_outcomes)
            curr_belief = list(np.round(bkt_model.curr_bel[:, 1], 4))
            kc_id = bkt_model.kcMap[topic_name]
            print("kc id: {}, current updated belief: ".format(kc_id))
            for index, bel in enumerate(curr_belief):
                if index == kc_id:
                    print("{} ".format(strGreen(bel)), end="")
                else:
                    print("{} ".format(bel), end="")
        bkt_model.updateReward()  # collect reward of curr. user into plan_model.finalRewards

    data["n_rounds"] = len(action)
    data["belief_context"] = np.array(belief_context)
    data["attempt_context"] = np.array(attempt_context)
    data["correct_context"] = np.array(correct_context)
    data["action"] = np.array(action)
    data["reward"] = np.array(reward)
    data["pscore"] = np.array(pscore)
    data["position"] = np.array(len(action) * [0])
    data["action_context"] = np.identity(n=n_actions)
    data["user_index_dict"] = user_index_dict
    data["eval_policy_action_dist"] = np.array(eval_policy_action_dist_list)

    pickle.dump(data, open("{}/logged_bandit_feedback_{}_action_dist.pkl".format(
        data_dir, policy), "wb"))


def load_bkt_model(reward_model, problem_seq):
    user_name_pretest_posttest_scores = pickle.load(
        open("../data/MasteryGrids/user_name_pretest_posttest_scores.pkl", "rb"))
    user_records_dict = pickle.load(open("../data/MasteryGrids/user_records_dict.pkl", "rb"))
    problem_topic_mapping = pickle.load(open("../data/MasteryGrids/problem_kc_mapping.pkl", "rb"))
    kcs = pickle.load(open("../data/MasteryGrids/kcs.pkl", "rb"))
    model_file_path = "../data/MasteryGrids/bkt_input_full_model.txt"
    O, T, pi, kcMap = getYudelsonModel(model_file_path, len(kcs), kcs)

    train_X = []
    train_y = []
    train_bel_X = []
    init_belief = pi[:, 1]
    for index, user in enumerate(user_name_pretest_posttest_scores):
        if user not in user_name_pretest_posttest_scores:
            continue  # train_users contains some users who do not have pre-post tests
        pretest, posttest = user_name_pretest_posttest_scores[user]
        print("{}: user: {}, pretest: {}, posttest: {}".format(
            index, user, pretest, posttest))
        records = []
        for user, question, topic, result, _ in user_records_dict[user]:
            records.append([question, result])
        last_belief = compute_bkt_last_belief(pi, O, T, records, kcMap, problem_topic_mapping)
        print("init belief: {}, {}".format(len(list(init_belief)), list(init_belief)))
        print("last belief: {}, {}".format(len(list(last_belief)), list(last_belief)))

        x = np.append([pretest], last_belief)
        train_X.append(x)
        train_bel_X.append(last_belief)
        if reward_model == "logistic":
            train_y.append(round(posttest))
        elif reward_model in ["linear", "ridge", "svr"]:
            train_y.append(posttest)
        else:
            raise AttributeError
    print("X shape: {}, y shape: {}".format(np.array(train_X).shape, np.array(train_y).shape))
    # reward model is logistic regression and the train_y should be binary
    # but if we binarize the train_y, it may become unbalanced data
    if reward_model == "logistic":
        planRewardModel = LogisticModel(train_X, train_y)
    elif reward_model == "linear":
        planRewardModel = LinearModel(train_X, train_y)
    elif reward_model == "ridge":
        planRewardModel = RidgeModel(train_X, train_y)
    elif reward_model == "svr":
        planRewardModel = SVRModel(train_X, train_y)
    else:
        raise AttributeError

    bkt_model = BKTModel(O, T, pi, kcMap, planRewardModel, problem_seq, problem_topic_mapping)
    return bkt_model


def load_pear_model(fold):
    seed = 1024
    arg_parser = argparse.ArgumentParser(description="DeepKT")
    arg_parser.add_argument('-c', '--config',
                            type=str, metavar='config_json_file',
                            default='../deepkt/configs/5folds/DKM/MasteryGrids/'
                                    'DKM_MasteryGrids_fold_{}.json'.format(fold),
                            help='the configuration json file')
    args = arg_parser.parse_args()
    config = process_config(args.config)
    print(config.checkpoint_file)
    config.max_seq_len = 1000
    config.seed = seed
    DKMAgent = globals()[config.agent]
    dkm_agent = DKMAgent(config)
    Q_matrix = dkm_agent.get_Q_matrix()
    return dkm_agent, Q_matrix


if __name__ == '__main__':
    # policy = "random"
    # policy = "instruct_seq"
    # policy = "inv_instruct_seq"
    # policy = "mastery"
    # policy = "highest_prob_correct"
    # policy = "myopic"
    policy = "pear"

    if policy == "pear":
        pear_model, Q_matrix = load_pear_model(fold=1)
    else:
        pear_model = None
        Q_matrix = None

    # We need the reward model for myopic policy
    # reward_model_type = "logistic"
    reward_model_type = "linear"
    print("offline evaluation policy: {}".format(policy))
    data_dir = "../data/MasteryGrids"

    problem_topic_mapping = pickle.load(open("{}/problem_kc_mapping.pkl".format(data_dir), "rb"))
    problem_seq = generate_instructional_sequence(problem_topic_mapping)
    bkt_model = load_bkt_model(reward_model_type, problem_seq)

    user_name_pretest_posttest_scores = pickle.load(
        open("{}/user_name_pretest_posttest_scores.pkl".format(data_dir), "rb"))
    user_records_dict = pickle.load(open("{}/user_records_dict.pkl".format(data_dir), "rb"))
    problem_name_problem_id_mapping = {}
    for i, name in enumerate(problem_seq):
        problem_name_problem_id_mapping[name] = i

    lenTraj = []
    pretestList = []
    users_q_list = []
    users_a_list = []
    users_name_list = []
    for index, user in enumerate(user_name_pretest_posttest_scores):
        users_name_list.append(user)
        traj_len = len(user_records_dict[user])
        q_list = []
        a_list = []
        for user, question, topic, result, _ in user_records_dict[user]:
            q_list.append(question)
            a_list.append(result)
        users_q_list.append(q_list)
        users_a_list.append(a_list)
        lenTraj.append(traj_len)
        pretest, posttest = user_name_pretest_posttest_scores[user]
        pretestList.append(pretest)
        print("{}: user: {}, traj len: {}, pretest: {}, posttest: {}".format(
            index, user, traj_len, pretest, posttest))
        print("question and answer pair: {}".format(list(zip(q_list, a_list))))
        print()
    eval_policy(bkt_model, policy, lenTraj, pretestList, users_q_list, users_a_list,
                users_name_list, pear_model=pear_model, Q_matrix=Q_matrix)
