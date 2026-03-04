import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        action_bounds=None,
        gripper_indices=None,
        gripper_open_prior=True,
        gripper_mask=False,
        sigma_scale=None,
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix

        # --- Per-Dimension Sigma-Skalierung ---
        # sigma_scale: (action_dim,) Tensor — Multiplikator auf var_scale pro Dim.
        # Ermöglicht z.B. mehr z-Exploration bei gleicher xy-Exploration
        # im normalisierten Raum. Kompensiert unterschiedliche action_std
        # ohne den CEM-Raum komplett umzubauen.
        # Default: None → alle Dims gleich (= bisheriges Verhalten)
        if sigma_scale is not None:
            self.sigma_scale = sigma_scale.float()
            print(f"  CEM Sigma-Scale: {self.sigma_scale[:min(8, action_dim)].tolist()} (erste {min(8, action_dim)} Dims)")
        else:
            self.sigma_scale = None

        # Action Bounds (im normalisierten Raum)
        # action_bounds: dict mit "lower" und "upper" als (action_dim,) Tensoren
        # → CEM-Samples werden auf diesen Bereich geclampt
        if action_bounds is not None:
            self.action_lower = action_bounds["lower"].float()
            self.action_upper = action_bounds["upper"].float()
            print(f"  CEM Action Bounds: lower={self.action_lower[:4].tolist()}, upper={self.action_upper[:4].tolist()} (erste 4 Dims)")
        else:
            self.action_lower = None
            self.action_upper = None

        # Gripper-Indices für binäre Quantisierung
        # Bei 8D Actions: Gripper-Dimensionen (Index 3, 7) sollen nur {0, 1} sein
        # In normalisiertem Raum: 0 → (0 - mean) / std, 1 → (1 - mean) / std
        self.gripper_indices = gripper_indices
        self.gripper_open_prior = gripper_open_prior

        # Gripper-Mask: Gripper-Dims werden komplett eingefroren (sigma=0).
        self.gripper_mask = gripper_mask
        if self.gripper_mask and self.gripper_indices:
            print(f"  CEM Gripper-Mask: AKTIV — Dims {self.gripper_indices} eingefroren (sigma=0)")
            if gripper_indices:
                print(f"    Gripper-Quantisierung implizit deaktiviert (Mask hat Vorrang)")

    def _clamp_actions(self, action):
        """Clampt Actions auf gültige Bounds (normalisierter Raum)."""
        if self.action_lower is not None:
            lower = self.action_lower.to(action.device)
            upper = self.action_upper.to(action.device)
            action = action.clamp(min=lower, max=upper)
        return action

    def _quantize_gripper(self, action):
        """Quantisiert Gripper-Dimensionen auf {norm(0), norm(1)} (nächster Wert).
        
        Gripper ist binär (offen=0, geschlossen=1). CEM sampelt aber kontinuierlich.
        → Snapping auf den nächsten gültigen normalisierten Wert reduziert den
        Suchraum und verhindert physikalisch unsinnige Gripper-Zwischenwerte.
        
        Wird bei gripper_mask=True NICHT aufgerufen (sigma=0 → kein Sampling).
        """
        if self.gripper_indices is None:
            return action
        if self.gripper_mask:
            return action  # Mask-Modus: Gripper ist fix, keine Quantisierung nötig

        action_mean = self.preprocessor.action_mean
        action_std = self.preprocessor.action_std

        for gi in self.gripper_indices:
            if gi < action.shape[-1]:
                norm_0 = (0.0 - action_mean[gi].item()) / action_std[gi].item()
                norm_1 = (1.0 - action_mean[gi].item()) / action_std[gi].item()
                mid = (norm_0 + norm_1) / 2.0
                action[..., gi] = torch.where(
                    action[..., gi] < mid,
                    torch.tensor(norm_0, device=action.device, dtype=action.dtype),
                    torch.tensor(norm_1, device=action.device, dtype=action.dtype),
                )
        return action

    def init_mu_sigma(self, obs_0, actions=None):
        """Initialisiert CEM mu und sigma im normalisierten Raum.
        
        mu = 0 (= action_mean in Weltkoordinaten).
        sigma = var_scale * sigma_scale (per-Dimension skalierbar).
        
        sigma_scale erlaubt gezielte Kompensation: z.B. wenn action_std_z < action_std_xy,
        dann sigma_scale_z > 1.0 → mehr physische z-Exploration.
        
        args:
            obs_0: dict mit "visual" und "proprio"
            actions: (B, T, action_dim) — Warm-Start Actions (normalisiert)
        """
        n_evals = obs_0["visual"].shape[0]

        # Basis-Sigma: var_scale * ones, optional per-Dim skaliert
        base_sigma = self.var_scale * torch.ones(self.action_dim)
        if self.sigma_scale is not None:
            base_sigma = base_sigma * self.sigma_scale
        sigma = base_sigma.unsqueeze(0).unsqueeze(0).expand(
            n_evals, self.horizon, self.action_dim
        ).clone()

        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)

            # Gripper Open-Prior: mu = norm(0) statt 0.0 (= Midpoint)
            if self.gripper_open_prior and self.gripper_indices is not None:
                action_mean = self.preprocessor.action_mean
                action_std = self.preprocessor.action_std
                for gi in self.gripper_indices:
                    if gi < self.action_dim:
                        norm_open = (0.0 - action_mean[gi].item()) / action_std[gi].item()
                        new_mu[:, :, gi] = norm_open

            mu = torch.cat([mu, new_mu.to(device)], dim=1)

        # Gripper-Mask: sigma=0 für Gripper-Dims
        if self.gripper_mask and self.gripper_indices is not None:
            for gi in self.gripper_indices:
                if gi < self.action_dim:
                    sigma[:, :, gi] = 0.0
                    if not self.gripper_open_prior:
                        action_mean = self.preprocessor.action_mean
                        action_std = self.preprocessor.action_std
                        norm_open = (0.0 - action_mean[gi].item()) / action_std[gi].item()
                        mu[:, :, gi] = norm_open

        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        CEM-Optimierung im normalisierten Raum.
        
        Args:
            obs_0: Aktuelle Beobachtung (dict: visual, proprio)
            obs_g: Ziel-Beobachtung
            actions: (B, T, action_dim) normalisierte Warm-Start Actions
        Returns:
            actions: (B, T, action_dim) normalisierte Actions
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # mu selbst als ein Sample

                # Action Bounds & Gripper-Quantisierung
                action = self._clamp_actions(action)
                action = self._quantize_gripper(action)

                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )

                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

                # mu auf Bounds clampen (verhindert Drift über Iterationen)
                mu[traj] = self._clamp_actions(mu[traj].unsqueeze(0)).squeeze(0)

                # Gripper-Mask: sigma wieder auf 0 setzen
                if self.gripper_mask and self.gripper_indices is not None:
                    for gi in self.gripper_indices:
                        if gi < self.action_dim:
                            sigma[traj, :, gi] = 0.0

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break

        return mu, np.full(n_evals, np.inf)
