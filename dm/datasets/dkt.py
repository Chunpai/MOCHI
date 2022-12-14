import numpy as np
from torch.utils.data import Dataset
from dm.datasets.transforms import SlidingWindow, Padding



class DKTDataset(Dataset):
    """
    prepare the data for data loader, including truncating long sequence and padding
    """

    def __init__(self, q_records, a_records, rec_q_records, rec_r_records, users, num_items,
                 max_seq_len, min_seq_len=2, stride=None, train=True, metric="auc"):
        """
        :param min_seq_len: used to filter out seq. less than min_seq_len
        :param max_seq_len: used to truncate seq. greater than max_seq_len
        :param max_subseq_len: used to truncate the lecture seq.
        """
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        if train and stride is not None:
            self.stride = stride
        else:
            self.stride = max_seq_len - 1
        self.metric = metric

        self.u_data, self.q_data, self.a_data, self.rec_q_data, self.rec_r_data = self._transform(
            q_records, a_records, rec_q_records, rec_r_records, users)
        print("train samples.: {}".format(self.q_data.shape))
        self.length = len(self.q_data)

    def __len__(self):
        """
        :return: the number of training samples rather than training users
        """
        return self.length

    def __getitem__(self, idx):
        """
        for DKT, we apply one-hot encoding and return the idx'th data sample here, we dont
        need to return target, since the input of DKT also contain the target information
        reference: https://github.com/jennyzhang0215/DKVMN/blob/master/code/python3/model.py

        for SAKT, we dont apply one-hot encoding, instead we return the past interactions,
        current exercises, and current answers
        # reference: https://github.com/TianHongZXY/pytorch-SAKT/blob/master/dataset.py

        idx: sample index
        """
        user = self.u_data[idx]
        questions = self.q_data[idx]
        answers = self.a_data[idx]
        rec_questions = self.rec_q_data[idx]
        rec_rewards = self.rec_r_data[idx]
        assert len(questions) == len(answers)

        # one_hot_interactions = np.zeros((self.max_seq_len, 2 * self.num_items + 1), dtype=int)
        # for i, q in enumerate(questions):
        #     label_index = q + answers[i] * self.num_items
        #     if label_index != 0:
        #         # note that, padding is a zero vector, and question id should starts from 1
        #         one_hot_interactions[i, label_index] = 1

        if self.metric == "rmse":
            interactions = []
            for i, q in enumerate(questions):
                interactions.append([q, answers[i]])
            interactions = np.array(interactions, dtype=float)
        else:
            interactions = np.zeros(self.max_seq_len, dtype=int)
            for i, q in enumerate(questions):
                interactions[i] = q + answers[i] * self.num_items

        target_answers = answers[1:]
        target_rewards = rec_rewards[1:]
        # target mask is used to select true labels
        target_mask = (questions[1:] != 0)
        pred_mask = np.full((self.max_seq_len - 1, self.num_items), False)
        pred_rec_mask = np.full((self.max_seq_len - 1, self.num_items), False)
        index_list = []
        question_list = []  # used to generate pred mask
        rec_question_list = []
        for i, (q, rec_q) in enumerate(zip(questions[1:], rec_questions[1:])):   # DKT only make
            # max_seq_len -1
            # predictions
            if q != 0:
                index_list.append(i)
                # output of DKT is vector with size self.num_items, that is why q-1
                question_list.append(q - 1)
                rec_question_list.append(rec_q - 1)
        pred_mask[index_list, question_list] = True
        pred_rec_mask[index_list, rec_question_list] = True
        # return one_hot_interactions, pred_mask, target_answers, target_mask
        return user, interactions, pred_mask, target_answers, target_mask, target_rewards, \
            pred_rec_mask

    def _transform(self, q_records, a_records, rec_q_records, rec_r_records, users):
        """
        transform the data into feasible input of model,
        truncate the seq. if it is too long and
        pad the seq. with 0s if it is too short
        """
        assert len(q_records) == len(a_records) == len(rec_q_records) == len(rec_r_records)
        u_data = []
        q_data = []
        a_data = []
        rec_q_data = []
        rec_r_data = []
        # if seq length is less than max_seq_len, the sliding will pad it with fillvalue
        # the reason of inserting the first attempt with 0 and setting stride=self.max_seq_len-1
        # is to make sure every test point will be evaluated if a model cannot test the first
        # attempt
        sliding = SlidingWindow(self.max_seq_len, self.stride, fillvalue=0)
        for index, u in enumerate(users):
            q_list = q_records[index]
            a_list = a_records[index]
            rec_q_list = rec_q_records[index]
            rec_r_list = rec_r_records[index]
            q_list.insert(0, 0)
            a_list.insert(0, 0)
            rec_q_list.insert(0, 0)
            rec_r_list.insert(0, 0)
            assert len(q_list) == len(a_list) == len(rec_q_list) == len(rec_r_list)
            sample = {"q": q_list, "a": a_list, "rec_q": rec_q_list, "rec_r": rec_r_list}
            output = sliding(sample)
            u_data.extend([u] * len(output["q"]))
            q_data.extend(output["q"])
            a_data.extend(output["a"])
            rec_q_data.extend(output["rec_q"])
            rec_r_data.extend(output["rec_r"])
        return np.array(u_data), np.array(q_data), np.array(a_data), np.array(rec_q_data), np.array(
            rec_r_data)
