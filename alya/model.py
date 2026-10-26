"""Actor-critic: MLP(наблюдение) -> GRU -> (смещение курсора, клавиша, ценность)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sim import GLOB_F, N_NEXT, OBJ_F, OBS_DIM

MAX_ACTION = 5.0  # |dx|,|dy| <= 5 * MOVE_SCALE px за шаг
COND_DIM = 2      # уровень игрока: точность и доля промахов
IN_DIM = OBS_DIM + COND_DIM
# старый формат входа (10 признаков на ноту, без пути слайдера) — модели до v4
LEGACY_OBJ_F = 10
LEGACY_IN_DIM = N_NEXT * LEGACY_OBJ_F + GLOB_F + COND_DIM
# клавиши — одно решение из 4: ничего / K1 / K2 / обе (независимые K1 и K2 давали даблтапы)
KEY_COMBOS = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def key_index(keys):
    """keys [..., 2] (0/1) -> индекс комбинации [...]."""
    return (keys[..., 0] + 2 * keys[..., 1]).long()


def skill_features(acc, miss_rate):
    """Условие «играй как игрок с такой точностью и долей промахов» -> признаки модели."""
    acc, miss_rate = np.asarray(acc, np.float32), np.asarray(miss_rate, np.float32)
    return np.stack([np.clip((acc - 0.95) / 0.05, -4, 1), np.clip(miss_rate * 20, 0, 4)], -1).astype(np.float32)


def with_cond(obs, cond):
    """Приписать к наблюдению [..., OBS_DIM] условие [..., COND_DIM] (numpy или torch)."""
    if isinstance(obs, torch.Tensor):
        cond = torch.as_tensor(cond, device=obs.device, dtype=obs.dtype).expand(*obs.shape[:-1], COND_DIM)
        return torch.cat([obs, cond], -1)
    return np.concatenate([obs, np.broadcast_to(cond, obs.shape[:-1] + (COND_DIM,))], -1)


class AlyaNet(nn.Module):
    def __init__(self, obs_dim: int = IN_DIM, hidden: int = 256):
        super().__init__()
        self.hidden, self.obs_dim = hidden, obs_dim
        # скорость курсора и состояние клавиш скрыты от модели: иначе BC учится копировать
        # своё прошлое действие («клавиша зажата — держи») и в замкнутом цикле залипает
        mask = torch.ones(obs_dim)
        g = N_NEXT * (LEGACY_OBJ_F if obs_dim == LEGACY_IN_DIM else OBJ_F)
        mask[[g + 0, g + 1, g + 4, g + 8]] = 0
        self.register_buffer("in_mask", mask)
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
        )
        self.gru = nn.GRU(256, hidden)
        self.mu = nn.Linear(hidden, 2)
        self.key = nn.Linear(hidden, 4)  # логиты комбинаций KEY_COMBOS
        self.value = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), -1.0))
        nn.init.orthogonal_(self.mu.weight, 0.1)
        nn.init.zeros_(self.mu.bias)

    def init_hidden(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden, device=device)

    def forward(self, obs, h, resets=None):
        """obs [T,B,D], h [1,B,H], resets [T,B] (1 = обнулить h перед шагом)."""
        if obs.shape[-1] == IN_DIM and self.obs_dim == LEGACY_IN_DIM:  # старая модель, новый сим
            per = obs[..., :N_NEXT * OBJ_F].unflatten(-1, (N_NEXT, OBJ_F))[..., :LEGACY_OBJ_F]
            obs = torch.cat([per.flatten(-2), obs[..., N_NEXT * OBJ_F:]], -1)
        x = self.enc(obs * self.in_mask)
        if resets is None:
            out, h = self.gru(x, h)
        else:
            outs = []
            for t in range(x.shape[0]):
                h = h * (1 - resets[t]).view(1, -1, 1)
                o, h = self.gru(x[t:t + 1], h)
                outs.append(o)
            out = torch.cat(outs)
        return self.mu(out), self.key(out), self.value(out).squeeze(-1), h

    def dists(self, mu, key_logit):
        std = self.log_std.clamp(-4, 1).exp().expand_as(mu)
        return torch.distributions.Normal(mu, std), torch.distributions.Categorical(logits=key_logit)

    @torch.no_grad()
    def act(self, obs, h, deterministic=False):
        """obs [B,D] -> move [B,2], keys [B,2], logp [B], value [B], h."""
        mu, kl, v, h = self(obs.unsqueeze(0), h)
        mu, kl, v = mu[0], kl[0], v[0]
        nd, kd = self.dists(mu, kl)
        if deterministic:
            move, ki = mu, kl.argmax(-1)
        else:
            move, ki = nd.sample(), kd.sample()
        logp = nd.log_prob(move).sum(-1) + kd.log_prob(ki)
        return move, KEY_COMBOS.to(ki.device)[ki], logp, v, h


def save(model: AlyaNet, path, **extra):
    torch.save({"state": model.state_dict(), "hidden": model.hidden, "obs_dim": model.obs_dim, **extra}, path)


def load(path, device="cpu") -> AlyaNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = AlyaNet(ck.get("obs_dim", OBS_DIM), ck.get("hidden", 256)).to(device)
    if m.obs_dim not in (IN_DIM, LEGACY_IN_DIM):
        raise SystemExit(f"{path}: старый формат входа ({m.obs_dim}), нужен {IN_DIM} — переобучите BC")
    m.load_state_dict(ck["state"])
    return m
