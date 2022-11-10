import pickle
from sklearn.model_selection import train_test_split, KFold
import random
import numpy as np


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


def get_data(data_str):
    data = pickle.load(open("../data/{}/data.pkl".format(data_str), "rb"))
    user_list = data["users"]
    item_list = data['items']
    user_records_dict = data["records"]
    pretest_posttest_dict = data["pretest_posttest_dict"]
    next_item_distr_dict = data["next_items_dict"]
    return user_list, item_list, user_records_dict, pretest_posttest_dict, next_item_distr_dict


def generate_bandit_feedback(user_records):
    next_q_dist = generate_problem_seq(user_records)

    problem_topic_mapping = pickle.load(
        open("../data/MasteryGrids/problem_kc_mapping.pkl", "rb"))
    problem_seq = generate_instructional_sequence(problem_topic_mapping)

    problem_name_id_mapping = {}
    for i, name in enumerate(problem_seq):
        problem_name_id_mapping[name] = i
    data = {}
    n_actions = len(problem_seq)
    data["n_actions"] = n_actions

    expected_reward_dict = {}
    for i in range(n_actions):
        if i not in expected_reward_dict:
            expected_reward_dict[i] = {}
        for j in range(n_actions):
            if j not in expected_reward_dict[i]:
                expected_reward_dict[i][j] = [0, 0]

    context = []
    action = []
    reward = []
    pscore = []
    user_list = list(user_records.keys())
    for user in user_list:
        questions = user_records[user]
        size = len(questions)
        for i in range(size - 1):
            curr_q = questions[i]
            next_q = questions[i + 1]
            x_i = problem_name_id_mapping[curr_q]
            context.append([x_i])
            x_j = problem_name_id_mapping[next_q]

            # index = problem_seq.index(curr_q)
            # if index + 1 != len(problem_seq):
            #     a_i = problem_name_id_mapping[problem_seq[index + 1]]
            #     sorted_items = sorted(next_q_dist[curr_q].items(), key=lambda x: x[1], reverse=True)
            #     propensity = sorted_items[0][1]
            # else:
            #     a_i = problem_name_id_mapping[problem_seq[0]]
            #     propensity = 1.

            if len(next_q_dist[curr_q]) == 0:
                a_i = problem_name_id_mapping[problem_seq[0]]
                propensity = 1.
            else:
                sorted_items = sorted(next_q_dist[curr_q].items(), key=lambda x: x[1], reverse=True)
                a_i = problem_name_id_mapping[sorted_items[0][0]]
                propensity = sorted_items[0][1]

            action.append(a_i)
            pscore.append(propensity)
            if x_j == a_i:
                r = 1
            else:
                r = 0
            reward.append(r)
            expected_reward_dict[x_i][a_i][r] += 1
    data["n_rounds"] = len(action)
    data["context"] = np.array(context)
    data["action"] = np.array(action)
    data["reward"] = np.array(reward)
    data["pscore"] = np.array(pscore)
    data["position"] = np.array(len(action) * [0])
    data["action_context"] = np.identity(n=n_actions)

    expected_reward_distr_dict = {}
    for x_i in expected_reward_dict:
        if x_i not in expected_reward_distr_dict:
            expected_reward_distr_dict[x_i] = [0] * n_actions
        for a_i in expected_reward_dict[x_i]:
            sum_reward = sum(expected_reward_dict[x_i][a_i])
            if sum_reward == 0:
                continue
            else:
                ratio = expected_reward_dict[x_i][a_i][1] / sum_reward
                expected_reward_distr_dict[x_i][a_i] = ratio
    expected_reward = []
    for x in context:
        expected_reward.append(expected_reward_distr_dict[x[0]])
    expected_reward = np.array(expected_reward)
    data["expected_reward"] = expected_reward

    pickle.dump(data, open("../data/MasteryGrids/all_logged_bandit_feedback.pkl", "wb"))


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


def generate_random_feedback(user_records):
    problem_topic_mapping = pickle.load(
        open("../data/MasteryGrids/problem_kc_mapping.pkl", "rb"))
    problem_seq = generate_instructional_sequence(problem_topic_mapping)

    problem_name_id_mapping = {}
    for i, name in enumerate(problem_seq):
        problem_name_id_mapping[name] = i
    data = {}
    n_actions = len(problem_seq)
    data["n_actions"] = n_actions

    user_index_dict = {}
    context = []
    action = []
    reward = []
    user_list = list(user_records.keys())
    print(user_list)
    for user in user_list:
        if user not in user_index_dict:
            user_index_dict[user] = []
        questions = user_records[user]
        size = len(questions)
        for i in range(size - 1):
            user_index_dict[user].append(len(context))  # map user to a list of context index
            curr_q = questions[i]
            next_q = questions[i + 1]
            x_i = problem_name_id_mapping[curr_q]
            context.append([x_i])
            x_j = problem_name_id_mapping[next_q]

            random_q = random.choice(problem_seq)
            a_i = problem_name_id_mapping[random_q]

            action.append(a_i)
            if x_j == a_i:
                r = 1
            else:
                r = 0
            reward.append(r)
    data["n_rounds"] = len(action)
    data["context"] = np.array(context)
    data["action"] = np.array(action)
    data["reward"] = np.array(reward)
    data["position"] = np.array(len(action) * [0])
    data["pscore"] = np.array(len(action) * [1. / n_actions])
    data["user_index_dict"] = user_index_dict

    pickle.dump(data, open("../data/MasteryGrids/random_bandit_feedback.pkl", "wb"))


def generate_instruct_seq_feedback(user_records):
    problem_topic_mapping = pickle.load(
        open("../data/MasteryGrids/problem_kc_mapping.pkl", "rb"))
    problem_seq = generate_instructional_sequence(problem_topic_mapping)

    problem_name_id_mapping = {}
    for i, name in enumerate(problem_seq):
        problem_name_id_mapping[name] = i
    data = {}
    n_actions = len(problem_seq)
    data["n_actions"] = n_actions

    user_index_dict = {}
    context = []
    action = []
    reward = []
    user_list = list(user_records.keys())
    for user in user_list:
        questions = user_records[user]
        size = len(questions)
        if user not in user_index_dict:
            user_index_dict[user] = []
        for i in range(size - 1):
            user_index_dict[user].append(len(context))  # map user to a list of context index
            curr_q = questions[i]
            next_q = questions[i + 1]
            x_i = problem_name_id_mapping[curr_q]
            context.append([x_i])
            x_j = problem_name_id_mapping[next_q]

            index = problem_seq.index(curr_q)
            if index + 1 != len(problem_seq):
                a_i = problem_name_id_mapping[problem_seq[index + 1]]
            else:
                a_i = problem_name_id_mapping[problem_seq[0]]

            action.append(a_i)
            if x_j == a_i:
                r = 1
            else:
                r = 0
            reward.append(r)
    data["n_rounds"] = len(action)
    data["context"] = np.array(context)
    data["action"] = np.array(action)
    data["reward"] = np.array(reward)
    data["position"] = np.array(len(action) * [0])
    data["pscore"] = np.array(len(action) * [1. / n_actions])
    data["user_index_dict"] = user_index_dict

    pickle.dump(data, open("../data/MasteryGrids/instruct_seq_bandit_feedback.pkl", "wb"))


def generate_inv_instruct_seq_feedback(user_records):
    problem_topic_mapping = pickle.load(
        open("../data/MasteryGrids/problem_kc_mapping.pkl", "rb"))
    problem_seq = generate_instructional_sequence(problem_topic_mapping)

    problem_name_id_mapping = {}
    for i, name in enumerate(problem_seq):
        problem_name_id_mapping[name] = i
    data = {}
    n_actions = len(problem_seq)
    data["n_actions"] = n_actions

    user_index_dict = {}
    context = []
    action = []
    reward = []
    user_list = list(user_records.keys())
    for user in user_list:
        questions = user_records[user]
        size = len(questions)
        if user not in user_index_dict:
            user_index_dict[user] = []
        for i in range(size - 1):
            user_index_dict[user].append(len(context))  # map user to a list of context index
            curr_q = questions[i]
            next_q = questions[i + 1]
            x_i = problem_name_id_mapping[curr_q]
            context.append([x_i])
            x_j = problem_name_id_mapping[next_q]

            index = problem_seq.index(curr_q)
            if index - 1 != -1:
                a_i = problem_name_id_mapping[problem_seq[index - 1]]
            else:
                a_i = problem_name_id_mapping[problem_seq[-1]]

            action.append(a_i)
            if x_j == a_i:
                r = 1
            else:
                r = 0
            reward.append(r)
    data["n_rounds"] = len(action)
    data["context"] = np.array(context)
    data["action"] = np.array(action)
    data["reward"] = np.array(reward)
    data["position"] = np.array(len(action) * [0])
    data["pscore"] = np.array(len(action) * [1. / n_actions])
    data["user_index_dict"] = user_index_dict

    pickle.dump(data, open("../data/MasteryGrids/inv_instruct_seq_bandit_feedback.pkl", "wb"))


if __name__ == '__main__':
    data_str = "MasteryGrids"
    users, items, user_records, pretest_posttest_dict, next_item_distr_dict = get_data(data_str)
    generate_bandit_feedback(user_records)
    generate_random_feedback(user_records)
    generate_instruct_seq_feedback(user_records)
    generate_inv_instruct_seq_feedback(user_records)
