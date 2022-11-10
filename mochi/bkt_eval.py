__author__ = "Chunpai W."
__email__ = "cwang25@albany.edu"

# policies are built based on BKT plan model, which is used to trace student's knowledge based on
# historical performance.


import argparse
import pickle
import numpy as np
from numpy.random import RandomState

from mochi.bkt import getYudelsonModel, compute_bkt_last_belief
from mochi.bkt import BKTModel
from mochi.rewards import LogisticModel, LinearModel
from mochi.utils import *
from scipy.stats import spearmanr

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


def eval_policy(plan_model, policy, traj_lengths, pretest_list, users_question_list,
                users_answer_list, users_name_list, next_questions_dict, top_k=1):

    data = {}
    n_actions = len(problem_seq)
    data["n_actions"] = n_actions

    prBlue("Policy: {}, Number of Test Users: {}".format(policy, len(users_name_list)))
    # if given a list of trajectory lengths, sample numTraj trajectory lengths with replacement.

    # sample a test user

    user_index_dict = {}
    context = []
    action = []
    reward = []
    for traj, user_name in enumerate(users_name_list):
        # reset the test student models' initial belief when new trajectory is sampled as the
        # training user's initial belief
        if user_name not in user_index_dict:
            user_index_dict[user_name] = []
        pretest = pretest_list[traj]
        user_q_list = users_question_list[traj]
        user_a_list = users_answer_list[traj]
        plan_model.resetState(pretest)  # for plan model, we don't reset the pretest score

        q_list = []
        a_list = []
        for t in range(traj_lengths[traj] if type(traj_lengths) == list else traj_lengths):
            # generate an action that follows policy (for BKT-MP, it is based on curr_bel)
            # once an action is generated, it is removed from the action space (problem_list)
            candidates = plan_model.plan(policy)
            question = candidates[0]
            if t + 1 < len(user_q_list):
                current_q = user_q_list[t]
                user_index_dict[user_name].append(len(context))
                context.append([problem_name_problem_id_mapping[current_q]])
                current_score = user_a_list[t]
                true_next_question = user_q_list[t + 1]
                action.append(problem_name_problem_id_mapping[question])
                if true_next_question in candidates:
                    r = 1
                else:
                    r = 0
                reward.append(r)

            true_q = user_q_list[t]
            true_a = user_a_list[t]
            topic_name = problem_topic_mapping[true_q]
            step_outcomes = [(true_q, true_a, 0)]
            q_list.append(true_q)
            a_list.append(true_a)
            plan_model.updateState(step_outcomes)
            curr_belief = list(np.round(plan_model.curr_bel[:, 1], 4))
            kc_id = plan_model.kcMap[topic_name]
            print("kc id: {}, current updated belief: ".format(kc_id))
            for index, bel in enumerate(curr_belief):
                if index == kc_id:
                    print("{} ".format(strGreen(bel)), end="")
                else:
                    print("{} ".format(bel), end="")
        plan_model.updateReward()  # collect reward of curr. user into plan_model.finalRewards

    data["n_rounds"] = len(action)
    data["context"] = np.array(context)
    data["action"] = np.array(action)
    data["reward"] = np.array(reward)
    data["position"] = np.array(len(action) * [0])
    data["pscore"] = np.array(len(action) * [1. / n_actions])
    data["user_index_dict"] = user_index_dict

    pickle.dump(data, open("../data/MasteryGrids/{}_bandit_feedback.pkl".format(policy), "wb"))


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


if __name__ == '__main__':
    fold = 1
    top_k = 1  # top-k questions for recommendation

    # policy = "random"
    # policy = "mastery"
    # policy = "highest_prob_correct"
    policy = "myopic"
    # policy = "baseline"

    # reward_model = "logistic"
    reward_model = "linear"
    print("offline evaluation policy: {}, top_k: {}".format(policy, top_k))
    data_dir = "../data/MasteryGrids"

    fold_users_dict = pickle.load(open("../data/MasteryGrids/fold_users_dict.pkl", "rb"))
    problem_topic_mapping = pickle.load(
        open("../data/MasteryGrids/problem_kc_mapping.pkl", "rb"))

    problem_seq = generate_instructional_sequence(problem_topic_mapping)

    user_name_pretest_posttest_scores = pickle.load(
        open("../data/MasteryGrids/user_name_pretest_posttest_scores.pkl", "rb"))
    user_records_dict = pickle.load(open("../data/MasteryGrids/user_records_dict.pkl", "rb"))
    next_questions_dict = pickle.load(open("{}/next_questions_dict.pkl".format(data_dir), "rb"))

    problem_name_problem_id_mapping = {}
    for i, name in enumerate(problem_seq):
        problem_name_problem_id_mapping[name] = i

    kcs = pickle.load(open("../data/MasteryGrids/kcs.pkl", "rb"))
    train_users_list = fold_users_dict[fold]["train"]
    test_users_list = fold_users_dict[fold]["test"]
    prBlue("train users: {} and test users: {}".format(len(train_users_list), len(test_users_list)))

    # use all training users' records to train the BKT model
    prBlue("loading trained BKT model")
    model_file_path = "../data/MasteryGrids/bkt_input_fold_{}_model.txt".format(fold)
    O, T, pi, kcMap = getYudelsonModel(model_file_path, len(kcs), kcs)
    # we need to make sure Shayan used same Yudelson Model as us
    # prior distribution: pi = [1-p(L_0), p(L_0)]
    # T is transition matrix:  is [[1-p(T), p(T)], [0, 1]]
    # O is observation matrix: is [[1-p(G), p(G)], [p(S), 1-p(S)]]
    # kcMap is topic name -> kc id

    train_bkt_X = []
    train_y = []
    init_belief = pi[:, 1]
    prBlue("train reward model based on pretest score and trained last belief from BKT")
    for index, user in enumerate(train_users_list):
        if user not in user_name_pretest_posttest_scores:
            continue
        pretest, posttest = user_name_pretest_posttest_scores[user]
        print("{}: user: {}, pretest: {}, posttest: {}".format(
            index, user, pretest, posttest))
        # print("init belief: {}".format(list(init_belief)))
        # print("init mastery: {}".format(init_mastery))
        records = []
        for user, question, topic, result, _ in user_records_dict[user]:
            records.append([question, result])
        last_belief = compute_bkt_last_belief(pi, O, T, records, kcMap, problem_topic_mapping)

        q_list = []
        a_list = []
        for user, question, topic, result, _ in user_records_dict[user]:
            q_id = problem_name_problem_id_mapping[question]
            q_list.append(q_id)
            a_list.append(result)

        print("init belief: {}".format(list(init_belief)))
        print("last belief: {}".format(list(last_belief)))
        print("")
        x = np.append([pretest], last_belief)
        train_bkt_X.append(x)
        if reward_model == "logistic":
            train_y.append(round(posttest))
        elif reward_model == "linear":
            train_y.append(posttest)
        else:
            raise AttributeError
    prBlue("plan model reward model input shape: {}, output shape: {}".format(
        np.array(train_bkt_X).shape, np.array(train_y).shape))
    # reward model is logistic regression and the train_y should be binary
    # but if we binarize the train_y, it may become unbalanced data
    if reward_model == "logistic":
        planRewardModel = LogisticModel(train_bkt_X, train_y)
    elif reward_model == "linear":
        planRewardModel = LinearModel(train_bkt_X, train_y)
    else:
        raise AttributeError

    # initialize the domain model which is used to simulate student's response
    problem_list = list(problem_topic_mapping.keys())
    prBlue("number of problem: {} and they are: {}".format(len(problem_list), problem_list))
    plan_model = BKTModel(O, T, pi, kcMap, planRewardModel, problem_list, problem_topic_mapping,
                          top_k)

    lenTraj = []
    pretestList = []
    prBlue("loading testing users' info")
    users_q_list = []
    users_a_list = []
    users_name_list = []
    users_list = list(train_users_list) + list(test_users_list)
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
    if policy != "random":
        eval_policy(plan_model, policy, lenTraj, pretestList, users_q_list, users_a_list,
                    users_name_list, next_questions_dict, top_k)
