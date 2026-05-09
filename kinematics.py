import torch
import torch.nn as nn
import numpy as np


class StephensonIII(nn.Module):
    def __init__(self):
        super(StephensonIII, self).__init__()
        self.L_raw = nn.Parameter(torch.tensor([
            1.0, 0.15, 0.8, 0.8, 0.5, 0.8, 0.8, 0.5
        ], dtype=torch.float32))
        self.X_fix = nn.Parameter(torch.tensor(1.2, dtype=torch.float32))
        self.X0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.Y0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def get_actual_lengths(self):
        return torch.nn.functional.softplus(self.L_raw) + 0.05

    def compute_residuals(self, theta2, theta_unk):
        L = self.get_actual_lengths()
        L1, L2, L3, L4, L5, L6, L7, alpha = L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7]
        t3, t4, t6, t7 = theta_unk[0], theta_unk[1], theta_unk[2], theta_unk[3]

        eq1 = L2 * torch.cos(theta2) + L3 * torch.cos(t3) - L4 * torch.cos(t4) - L1
        eq2 = L2 * torch.sin(theta2) + L3 * torch.sin(t3) - L4 * torch.sin(t4)
        eq3 = L2 * torch.cos(theta2) + L5 * torch.cos(t3 + alpha) + L6 * torch.cos(t6) - L7 * torch.cos(t7) - self.X_fix
        eq4 = L2 * torch.sin(theta2) + L5 * torch.sin(t3 + alpha) + L6 * torch.sin(t6) - L7 * torch.sin(t7)

        return torch.stack([eq1, eq2, eq3, eq4])

    def forward(self, theta2_seq):
        num_points = len(theta2_seq)
        foot_x_seq, foot_y_seq = [], []
        theta_unk_seq = []
        current_guess = torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float32)

        for i in range(num_points):
            theta2 = theta2_seq[i]

            with torch.no_grad():
                theta_unk = current_guess.clone()
                for _ in range(50):
                    theta_unk.requires_grad_(True)
                    F = self.compute_residuals(theta2, theta_unk)

                    if torch.max(torch.abs(F)) < 1e-5:
                        theta_unk = theta_unk.detach()
                        break

                    J = torch.autograd.functional.jacobian(
                        lambda x: self.compute_residuals(theta2, x), theta_unk
                    )
                    try:
                        delta = torch.linalg.solve(J, F)
                    except RuntimeError:
                        break
                    theta_unk = (theta_unk - delta).detach()

            theta_unk_differentiable = theta_unk.clone().requires_grad_(True)
            F_diff = self.compute_residuals(theta2, theta_unk_differentiable)
            J_diff = torch.autograd.functional.jacobian(
                lambda x: self.compute_residuals(theta2, x), theta_unk_differentiable
            )
            theta_out = theta_unk_differentiable - torch.linalg.solve(J_diff, F_diff)

            current_guess = theta_out.detach()
            theta_unk_seq.append(theta_out)

            t3_out, t6_out = theta_out[0], theta_out[2]
            L = self.get_actual_lengths()
            alpha_val = L[7]

            foot_x = self.X0 + L[1] * torch.cos(theta2) + L[5] * torch.cos(t3_out + alpha_val) + L[6] * torch.cos(
                t6_out)
            foot_y = self.Y0 + L[1] * torch.sin(theta2) + L[5] * torch.sin(t3_out + alpha_val) + L[6] * torch.sin(
                t6_out)

            foot_x_seq.append(foot_x)
            foot_y_seq.append(foot_y)

        return torch.stack(foot_x_seq), torch.stack(foot_y_seq), torch.stack(theta_unk_seq)