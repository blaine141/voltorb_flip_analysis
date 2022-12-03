import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from typing import Tuple, OrderedDict, List
from pytorch_lightning import LightningModule
from torch.optim import Adam, Optimizer
from environment import VoltorbFlipEnv
from agent import Agent
from dataset import RLDataset, ReplayBuffer
from tqdm import tqdm

class VoltorbFlipDQN(nn.Module):
    """Simple MLP network."""

    def __init__(self, hidden_size = 32):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, hidden_size, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(25 * hidden_size, 25),
        )

    def forward(self, x):
        return self.net(x.float())



class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
        self,
        batch_size: int = 16,
        lr: float = 1e-4,
        gamma: float = 0.99,
        sync_rate: int = 100,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 200,
        env = None,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
        """
        super().__init__()
        self.save_hyperparameters()

        if env is None:
            self.env = VoltorbFlipEnv()
        else:
            self.env = env

        self.net = VoltorbFlipDQN()
        self.target_net = VoltorbFlipDQN()

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_num_moves = 0
        self.num_moves = 0
        self.populate(self.hparams.warm_start_size)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in tqdm(range(steps)):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_qs = self.target_net(next_states)
            next_state_qs[:,:25][next_states[:,0,].flatten(1) == -1] = -1000
            next_state_values = next_state_qs.max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.L1Loss()(state_action_values, expected_state_action_values)

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = self.get_epsilon(self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame)
        self.log("epsilon", epsilon)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.log("reward", reward, on_step=True)

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_num_moves = self.num_moves
            self.num_moves = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "reward": reward,
                "train_loss": loss,
            }
        )
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"

