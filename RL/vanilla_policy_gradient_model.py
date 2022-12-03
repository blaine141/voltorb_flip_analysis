import argparse
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, optim
from torch.nn.functional import log_softmax, softmax
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch import nn

from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.models.rl.common.agents import PolicyAgent
from pl_bolts.models.rl.common.networks import MLP
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _GYM_AVAILABLE:
    import gym
else:  # pragma: no cover
    warn_missing_pkg("gym")


class Conv1x1(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(5, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        logits[x[:,:1] == -1] = -50
        return nn.Flatten()(logits)


@under_review()
class VanillaPolicyGradient(LightningModule):
    r"""PyTorch Lightning implementation of `Vanilla Policy Gradient`_.

    Paper authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour

    Model implemented by:

        - `Donal Byrne <https://github.com/djbyrne>`

    Example:
        >>> from pl_bolts.models.rl.vanilla_policy_gradient_model import VanillaPolicyGradient
        ...
        >>> model = VanillaPolicyGradient("CartPole-v0")

    Train::

        trainer = Trainer()
        trainer.fit(model)

    Note:
        This example is based on:
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter11/04_cartpole_pg.py

    Note:
        Currently only supports CPU and single GPU training with `accelerator=dp`

    .. _`Vanilla Policy Gradient`:
        https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf
    """

    def __init__(
        self,
        env: str,
        gamma: float = 0.99,
        lr: float = 0.01,
        batch_size: int = 8,
        n_steps: int = 10,
        avg_reward_len: int = 100,
        entropy_beta: float = 0.01,
        epoch_len: int = 1000,
        val_games: int = 2000,
        **kwargs
    ) -> None:
        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lr: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            batch_episodes: how many episodes to rollout for each batch of training
            entropy_beta: dictates the level of entropy per batch
            avg_reward_len: how many episodes to take into account when calculating the avg reward
            epoch_len: how many batches before pseudo epoch
        """
        super().__init__()

        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("This Module requires gym environment which is not installed yet.")

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.games_per_epoch = epoch_len
        self.entropy_beta = entropy_beta
        self.gamma = gamma
        self.n_steps = n_steps
        self.val_games = val_games

        self.save_hyperparameters()

        # Model components
        self.env_name = env
        self.env = gym.make(self.env_name)
        self.net = Conv1x1(self.env.observation_space.shape[0])
        self.agent = PolicyAgent(self.net)

        # Tracking metrics
        self.total_rewards = []
        self.episode_rewards = []
        self.done_episodes = 0
        self.avg_rewards = 0
        self.avg_reward_len = avg_reward_len
        self.eps = np.finfo(np.float32).eps.item()
        self.batch_states = []
        self.batch_actions = []

        self.state = self.env.reset()

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def train_batch(
        self,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.

        Returns:
            yields a tuple of Lists containing tensors for states, actions and rewards of the batch.
        """

        game_num = 0
        while game_num < self.games_per_epoch:

            action = self.get_action(self.state)

            next_state, reward, done, _ = self.env.step(action)

            self.episode_rewards.append(reward)
            self.batch_actions.append(action)
            self.batch_states.append(self.state)
            self.state = next_state

            if done:
                game_num += 1
                self.done_episodes += 1
                self.state = self.env.reset()
                self.total_rewards.append(sum(self.episode_rewards))
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

                returns = self.compute_returns(self.episode_rewards)

                for idx in range(len(self.batch_actions)):
                    yield self.batch_states[idx], self.batch_actions[idx], returns[idx]

                self.batch_states = []
                self.batch_actions = []
                self.episode_rewards = []

    def val_batch(
        self,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.

        Returns:
            yields a tuple of Lists containing tensors for states, actions and rewards of the batch.
        """

        deterministic_env = gym.make(self.env_name)
        deterministic_state = deterministic_env.reset()
        non_deterministic_env = deterministic_env.copy()
        non_deterministic_state = deterministic_state   

        non_deterministic_done = False
        deterministic_done = False

        game_num = 0
        while game_num < self.val_games:

            # Non-deterministic action
            if not non_deterministic_done:
                action = self.get_action(non_deterministic_state, deterministic=False)
                non_deterministic_state, non_deterministic_reward, non_deterministic_done, _ = non_deterministic_env.step(action)

            # Deterministic action
            if not deterministic_done:
                action = self.get_action(deterministic_state, deterministic=True)
                deterministic_state, deterministic_reward, deterministic_done, _ = deterministic_env.step(action[0])

            if non_deterministic_done and deterministic_done:
                game_num += 1
                deterministic_state = deterministic_env.reset()
                non_deterministic_env = deterministic_env.copy()
                non_deterministic_state = deterministic_state   

                non_deterministic_done = False
                deterministic_done = False

                yield non_deterministic_reward > 0, deterministic_reward > 0


    def compute_returns(self, rewards):
        """Calculate the discounted rewards of the batched rewards.

        Args:
            rewards: list of batched rewards

        Returns:
            list of discounted rewards
        """
        reward = 0
        returns = []

        for r in rewards[::-1]:
            reward = r + self.gamma * reward
            returns.insert(0, reward)

        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + self.eps)

        return returns

    def loss(self, states, actions, scaled_rewards) -> Tensor:
        """Calculates the loss for VPG.

        Args:
            states: batched states
            actions: batch actions
            scaled_rewards: batche Q values

        Returns:
            loss for the current batch
        """

        logits = self.net(states)

        # policy loss
        log_prob = log_softmax(logits, dim=1)
        log_prob_actions = scaled_rewards * log_prob[range(states.shape[0]), actions[0]]
        policy_loss = -log_prob_actions.mean()

        # entropy loss
        prob = softmax(logits, dim=1)
        entropy = -(prob * log_prob).sum(dim=1).mean()
        entropy_loss = -self.entropy_beta * entropy

        # total loss
        loss = policy_loss + entropy_loss
        self.log("train_loss", {"loss": loss, "policy_loss": policy_loss, "entropy_loss": entropy_loss})

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """
        states, actions, scaled_rewards = batch

        loss = self.loss(states, actions, scaled_rewards)

        START_ENTROPY_DECAY = 10000
        if self.done_episodes > START_ENTROPY_DECAY:
            self.entropy_beta = max(self.entropy_beta - 0.00002, 0)


        self.log("avg_reward", self.avg_rewards, on_step=True, on_epoch=True)
        self.log("entropy_beta", self.entropy_beta, on_step=True, on_epoch=True)

        log = {
            "episodes": self.done_episodes,
            "reward": self.total_rewards[-1],
            "avg_reward": self.avg_rewards,
        }
        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_rewards,
                "log": log,
                "progress_bar": log,
            }
        )

    def validation_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """
        non_deterministic_reward, deterministic_reward = batch

        self.log("deterministic_win_rate", deterministic_reward, on_step=False, on_epoch=True)
        self.log("non_deterministic_win_rate", non_deterministic_reward, on_step=False, on_epoch=True)


    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def _dataloader(self, batch_fn, batch_size) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(batch_fn)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader(self.train_batch, self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Get validation loader."""
        return self._dataloader(self.val_batch, 1)

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0][0][0].device.index if self.on_gpu else "cpu"

    def get_action(self, state: Tensor, deterministic=False) -> int:
        """Get the action to take in the current state.

        Args:
            state: current state"""
        if deterministic:
            states = torch.tensor([state], device=self.device)
            logits = self.net(states).squeeze(dim=-1)
            action = torch.argmax(logits, dim=-1)
        else:
            action = self.agent(state, self.device)[0]
        return action

    # @staticmethod
    # def add_model_specific_args(arg_parser) -> argparse.ArgumentParser:
    #     """Adds arguments for DQN model.

    #     Note:
    #         These params are fine tuned for Pong env.

    #     Args:
    #         arg_parser: the current argument parser to add to

    #     Returns:
    #         arg_parser with model specific cargs added
    #     """

    #     arg_parser.add_argument("--entropy_beta", type=float, default=0.01, help="entropy value")
    #     arg_parser.add_argument("--batches_per_epoch", type=int, default=10000, help="number of batches in an epoch")
    #     arg_parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    #     arg_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    #     arg_parser.add_argument("--env", type=str, required=True, help="gym environment tag")
    #     arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    #     arg_parser.add_argument("--seed", type=int, default=123, help="seed for training run")

    #     arg_parser.add_argument(
    #         "--avg_reward_len",
    #         type=int,
    #         default=100,
    #         help="how many episodes to include in avg reward",
    #     )

    #     return arg_parser
