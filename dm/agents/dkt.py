import copy

import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
import sys
from dm.datasets.transforms import SlidingWindow, Padding

from dm.agents.base import BaseAgent
from dm.graphs.models.dkt import DKT

# should not remove import statements below, it;s being used seemingly.
from dm.dataloaders import *

cudnn.benchmark = True
from dm.utils.misc import print_cuda_statistics
import warnings

warnings.filterwarnings("ignore")


class DKTAgent(BaseAgent):
    def __init__(self, config):
        """initialize the agent with provided config dict which inherent from the base agent
        class"""
        super().__init__(config)
        # initialize the data_loader, which include preprocessing the data
        data_loader = globals()[config.data_loader]  # remember to import the dataloader
        self.data_loader = data_loader(config=config)
        # self.data_loader have attributes: train_data, train_loader, test_data, test_loader
        # note that self.data_loader.train_data is same as self.data_loader.train_loader.dataset
        self.mode = config.mode
        self.metric = config.metric

        # config.input_dim = self.data_loader.num_items * 2 + 1
        config.output_dim = self.data_loader.num_items
        self.model = DKT(config)

        self.criterion = nn.BCELoss(reduction='sum')
        if config.optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.config.learning_rate,
                                       momentum=self.config.momentum)
        else:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.config.learning_rate,
                                        eps=self.config.epsilon)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=0,
            min_lr=1e-5,
            factor=0.5,
            verbose=True
        )

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            # torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        # this loading should be after checking cuda
        self.load_checkpoint(self.config.checkpoint_file)

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1
            if self.early_stopping():
                break

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        self.logger.info("\n")
        self.logger.info("Train Epoch: {}".format(self.current_epoch))
        self.logger.info("learning rate: {}".format(self.optimizer.param_groups[0]['lr']))
        self.train_loss = 0
        train_elements = 0
        for batch_idx, data in enumerate(tqdm(self.data_loader.train_loader)):
            interactions, pred_mask, target_answers, target_mask = data
            interactions = interactions.to(self.device)
            pred_mask = pred_mask.to(self.device)
            target_answers = target_answers.to(self.device)
            target_mask = target_mask.to(self.device)
            self.optimizer.zero_grad()  # clear previous gradient
            # need to double check the target mask
            output = self.model(interactions)
            label = torch.masked_select(target_answers, target_mask)
            output = torch.masked_select(output[:, :-1, :], pred_mask)
            loss = self.criterion(output.float(), label.float())
            self.train_loss += loss.item()
            train_elements += target_mask.int().sum()
            loss.backward()  # compute the gradient
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()  # update the weight
            self.current_iteration += 1
        # self.logger.info("Train Loss: {:.6f}".format(loss.item()))
        # self.logger.info("Train Loss: {:.6f}".format(train_loss / train_elements))
        self.train_loss = self.train_loss / train_elements
        self.scheduler.step(self.train_loss)
        self.logger.info("Train Loss: {:.6f}".format(self.train_loss))

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        if self.mode == "train":
            self.logger.info("Validation Result at Epoch: {}".format(self.current_epoch))
        else:
            self.logger.info("Test Result at Epoch: {}".format(self.current_epoch))
        test_loss = 0
        pred_labels = []
        true_labels = []
        with torch.no_grad():
            for data in self.data_loader.test_loader:
                interactions, pred_mask, target_answers, target_mask = data
                interactions = interactions.to(self.device)
                pred_mask = pred_mask.to(self.device)
                target_answers = target_answers.to(self.device)
                target_mask = target_mask.to(self.device)
                output = self.model(interactions)
                output = torch.masked_select(output[:, :-1, :], pred_mask)
                label = torch.masked_select(target_answers, target_mask)
                test_loss += self.criterion(output.float(), label.float()).item()
                pred_labels.extend(output.tolist())
                true_labels.extend(label.tolist())
        self.track_best(true_labels, pred_labels)

    def predict(self, q_list, a_list):
        # padding all zeros on the left
        new_q_list = q_list.copy()
        new_a_list = a_list.copy()
        padding = Padding(output_size=self.data_loader.max_seq_len, side="right", fillvalue=0)
        with torch.no_grad():
            new_q_list.insert(0, 0)
            new_a_list.insert(0, 0)
            new_q_list = new_q_list[-self.data_loader.max_seq_len:]
            new_a_list = new_a_list[-self.data_loader.max_seq_len:]
            assert len(new_q_list) == len(new_a_list)
            sample = {"q": new_q_list, "a": new_a_list}
            outputs = padding(sample)
            questions = outputs["q"]
            answers = outputs["a"]
            assert len(questions) == len(answers)

            if self.metric == "rmse":
                interactions = []
                for i, q in enumerate(questions):
                    interactions.append([q, answers[i]])
                interactions = np.array(interactions, dtype=float)
            else:
                interactions = np.zeros(self.data_loader.max_seq_len, dtype=int)
                for i, q in enumerate(questions):
                    interactions[i] = q + answers[i] * self.data_loader.num_items
            interactions = torch.from_numpy(np.array([interactions]))

            interactions = interactions.to(self.device)
            output = self.model(interactions)
            return output

    def init_plan_model(self, pretest, top_k, problem_id_problem_name_mapping, reward_model):
        self.pretest = pretest
        self.top_k = top_k
        self.reward = None
        self.problem_id_problem_name_mapping = problem_id_problem_name_mapping
        self.reward_model = reward_model
        self.curr_q_list = []
        self.curr_a_list = []
        dkt_outputs = self.predict(self.curr_q_list, self.curr_a_list)
        self.curr_mastery = dkt_outputs[0, 0, :].tolist()

    def update_state(self, q_list, a_list):
        self.curr_q_list = q_list
        self.curr_a_list = a_list
        dkt_outputs = self.predict(q_list, a_list)
        if len(q_list) >= self.data_loader.max_seq_len:
            self.curr_mastery = dkt_outputs[0, -1, :].tolist()
        else:
            self.curr_mastery = dkt_outputs[0, len(q_list), :]

    def update_reward(self):
        test_X = np.append(self.pretest, self.curr_mastery).reshape(1, -1)
        reward = self.reward_model.predict(test_X)[-1]
        if reward > 1:
            reward = 1
        elif reward < 0:
            reward = 0
        self.finalRewards.append(reward)

    def plan(self, policy):
        """return question names"""
        action = self.policies[policy]()
        return action

    def random_policy(self):
        num_list = random.sample(range(1, self.data_loader.num_items + 1), self.top_k)
        actions = []
        for problem_id in num_list:
            problem_name = self.problem_id_problem_name_mapping[problem_id]
            actions.append(problem_name)
        return actions

    def mastery_policy(self):
        scores = {}
        for problem in range(1, self.data_loader.num_items + 1):
            scores[problem] = 0.
            if self.curr_mastery[problem - 1] < self.mastery_threshold:
                scores[problem] += self.mastery_threshold - self.curr_mastery[problem - 1]
        sorted_scores = sorted(scores.keys(), key=lambda x: scores[x])
        actions = []
        for qid in sorted_scores[-self.top_k:][::-1]:
            actions.append(self.problem_id_problem_name_mapping[qid])
        return actions

    def myopic_policy(self, num_samples=1):
        scores = {}
        for problem in range(1, self.data_loader.num_items + 1):
            scores[problem] = 0.
            corr_prob = self.curr_mastery[problem - 1]
            incorr_prob = 1. - corr_prob
            p = np.array([incorr_prob, corr_prob])
            p /= p.sum()
            for sample in range(num_samples):
                # print(corr_prob, 1. -corr_prob)
                a = np.random.choice(2, p=p)
                q_list = self.curr_q_list + [problem]
                a_list = self.curr_a_list + [a]
                dkt_outputs = self.predict(q_list, a_list)
                if len(q_list) >= self.data_loader.max_seq_len:
                    tmp_mastery = dkt_outputs[0, -1, :].tolist()
                else:
                    tmp_mastery = dkt_outputs[0, len(q_list), :]
                tmp_X = np.append(self.pretest, tmp_mastery).reshape(1, -1)
                scores[problem] += self.reward_model.predict(tmp_X)[-1]
            scores[problem] /= num_samples
        sorted_scores = sorted(scores.keys(), key=lambda x: scores[x])
        actions = []
        for qid in sorted_scores[-self.top_k:][::-1]:
            actions.append(self.problem_id_problem_name_mapping[qid])
        return actions

    def highest_prob_correct_policy(self):
        scores = {}
        for problem in range(1, self.data_loader.num_items + 1):
            scores[problem] = self.curr_mastery[problem - 1]
        sorted_scores = sorted(scores.keys(), key=lambda x: scores[x])
        actions = []
        for qid in sorted_scores[-self.top_k:][::-1]:
            actions.append(self.problem_id_problem_name_mapping[qid])
        return actions
