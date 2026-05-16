from __future__ import annotations

from typing import Any, ClassVar, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

SelfMiniBatchREINFORCE = TypeVar("SelfMiniBatchREINFORCE", bound="MiniBatchREINFORCE")


class MiniBatchREINFORCE(OnPolicyAlgorithm):
    """Mini-batch REINFORCE nach Algorithmus 32 (Skript S. 168).

    Sammelt pro Update K vollständige Episoden, berechnet den
    Policy-Gradienten als Monte-Carlo-Mittel über die K Trajektorien mit
    reward-to-go als Q-Schätzer, und führt einen Gradient-Ascent-Schritt aus.

    Kein Critic, kein Bootstrapping, keine Advantage-Estimation. Die
    Implementation lehnt sich strukturell an SB3 A2C an, ersetzt aber den
    n_steps-Rollout-Buffer durch einen episodischen Sammler und entfernt
    den Value-Loss.

    Setzt n_envs=1 voraus (episodische Sammlung über eine Env-Kopie).

    Hinweis zu MPS: Auf kleinen Netzwerken (CartPole, Acrobot) ist CPU oft
    schneller als MPS, da der Transfer-Overhead die Rechenzeit dominiert.
    device="auto" bevorzugt MPS auf Apple Silicon — bei Laufzeitproblemen
    explizit device="cpu" setzen.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_episodes_per_update: int = 10,  # K in Algorithmus 32
        gamma: float = 0.99,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        normalize_returns: bool = True,
        use_rms_prop: bool = False,
        rms_prop_eps: float = 1e-5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=1,  # Dummy; collect_rollouts überschreibt episodische Sammlung
            gamma=gamma,
            gae_lambda=1.0,
            ent_coef=ent_coef,
            vf_coef=0.0,  # kein Value-Loss
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.n_episodes_per_update = n_episodes_per_update
        self.normalize_returns = normalize_returns
        self._episodic_buffer: list[dict] = []
        self._n_updates = 0

        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(
                alpha=0.99, eps=rms_prop_eps, weight_decay=0
            )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Sammelt n_episodes_per_update vollständige Episoden (Algorithmus 32, Skript S. 168).

        Ignoriert n_rollout_steps. Speichert Trajektorien in self._episodic_buffer.
        Setzt n_envs=1 voraus.
        """
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        rollout_buffer.reset()
        callback.on_rollout_start()

        self._episodic_buffer = []
        obs = self._last_obs  # shape (n_envs, obs_dim)

        for _ in range(self.n_episodes_per_update):
            ep: dict[str, list] = {"obs": [], "actions": [], "rewards": []}
            done = False

            while not done:
                obs_tensor = obs_as_tensor(obs, self.device)
                with th.no_grad():
                    actions, _values, _log_probs = self.policy(obs_tensor)
                actions_np = actions.cpu().numpy()

                clipped = actions_np
                if isinstance(self.action_space, spaces.Box):
                    if not self.policy.squash_output:
                        clipped = np.clip(
                            actions_np, self.action_space.low, self.action_space.high
                        )

                new_obs, rewards, dones, infos = env.step(clipped)
                self.num_timesteps += env.num_envs

                ep["obs"].append(obs[0].copy())
                ep["actions"].append(actions_np[0])
                ep["rewards"].append(float(rewards[0]))

                callback.update_locals(locals())
                if not callback.on_step():
                    self._last_obs = new_obs
                    return False

                self._update_info_buffer(infos, dones)
                obs = new_obs
                done = bool(dones[0])

            self._episodic_buffer.append(ep)

        self._last_obs = obs
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """Policy-Gradient-Update über alle gesammelten Episoden.

        Berechnet reward-to-go G_t^i rückwärts und optimiert den
        Policy-Gradienten (Algorithmus 32, Skript S. 168).
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        all_obs: list = []
        all_actions: list = []
        all_returns: list = []

        for ep in self._episodic_buffer:
            rewards = ep["rewards"]
            T = len(rewards)
            G = np.zeros(T, dtype=np.float32)
            g = 0.0
            # reward-to-go rückwärts berechnen: G_t = r_{t+1} + γ · G_{t+1}
            for t in reversed(range(T)):
                g = rewards[t] + self.gamma * g
                G[t] = g
            all_obs.extend(ep["obs"])
            all_actions.extend(ep["actions"])
            all_returns.extend(G.tolist())

        obs_tensor = obs_as_tensor(np.array(all_obs), self.device)

        if isinstance(self.action_space, spaces.Discrete):
            actions_tensor = th.tensor(
                np.array(all_actions, dtype=np.int64), device=self.device
            ).flatten()
        else:
            actions_tensor = th.tensor(
                np.array(all_actions, dtype=np.float32), device=self.device
            )

        returns_tensor = th.tensor(all_returns, dtype=th.float32, device=self.device)

        if self.normalize_returns:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # Forward-Pass unter aktueller Policy (Algorithmus 32, Skript S. 168, Update-Zeile 5)
        _values, log_prob, entropy = self.policy.evaluate_actions(
            obs_tensor, actions_tensor
        )

        policy_loss = -(log_prob * returns_tensor).mean()

        if entropy is None:
            entropy_loss = th.mean(log_prob.detach())  # Näherung
        else:
            entropy_loss = -entropy.mean()

        loss = policy_loss + self.ent_coef * entropy_loss

        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1

        ep_lengths = [len(ep["rewards"]) for ep in self._episodic_buffer]
        ep_returns_raw = [sum(ep["rewards"]) for ep in self._episodic_buffer]

        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record(
            "train/entropy",
            -entropy_loss.item() if entropy is not None else float("nan"),
        )
        self.logger.record("train/mean_return_G", float(np.mean(ep_returns_raw)))
        self.logger.record("train/std_return_G", float(np.std(ep_returns_raw)))
        self.logger.record("train/episodes_in_batch", len(self._episodic_buffer))
        self.logger.record("train/mean_episode_length", float(np.mean(ep_lengths)))
        self.logger.record(
            "train/learning_rate",
            self.lr_schedule(self._current_progress_remaining),
        )

    def learn(
        self: SelfMiniBatchREINFORCE,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "MiniBatchREINFORCE",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfMiniBatchREINFORCE:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
