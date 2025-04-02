import numpy as np
import torch
import torch.cuda
import torch.nn as nn
from data_enhancement import data_enhancement
from pymoo.indicators.hv import HV


def gen_offspring(Parent, DV, surrogate_model, boundary):
    rows_to_take = int(1 / 3 * Parent.shape[0])
    pop = Parent[:rows_to_take, :]

    if len(pop) % 2 == 1:
        pop = pop[:-1]

    DV = int(DV)

    augmentation_factor = 10
    enhanced_pop = data_enhancement(pop, augmentation_factor)
    dataset = torch.tensor(enhanced_pop).float()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = dataset.to(device)

    num_steps = 25
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-1 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    assert (
        alphas.shape
        == alphas_prod.shape
        == alphas_prod_p.shape
        == alphas_bar_sqrt.shape
        == one_minus_alphas_bar_log.shape
        == one_minus_alphas_bar_sqrt.shape
    )

    betas = betas.to(device)
    alphas_prod = alphas_prod.to(device)
    alphas_prod_p = alphas_prod_p.to(device)
    alphas_bar_sqrt = alphas_bar_sqrt.to(device)
    one_minus_alphas_bar_log = one_minus_alphas_bar_log.to(device)
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)

    def q_x(x_0, t):
        """Get x[t] at any time t based on x[0]"""
        noise = torch.randn_like(x_0).to(device)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return alphas_t * x_0 + alphas_1_m_t * noise

    class MLPDiffusion(nn.Module):

        def __init__(self, n_step, num_units=128):
            super(MLPDiffusion, self).__init__()

            self.linears = nn.ModuleList(
                [
                    nn.Linear(DV, num_units),
                    nn.ReLU(),
                    nn.Linear(num_units, num_units),
                    nn.ReLU(),
                    nn.Linear(num_units, num_units),
                    nn.ReLU(),
                    nn.Linear(num_units, DV),
                ]
            )
            self.step_embeddings = nn.ModuleList(
                [
                    nn.Embedding(n_step, num_units),
                    nn.Embedding(n_step, num_units),
                ]
            )

        def forward(self, x_0, t):

            x = x_0
            for idx, embedding_layer in enumerate(self.step_embeddings):
                t_embedding = embedding_layer(t)
                x = self.linears[2 * idx](x)
                x += t_embedding
                x = self.linears[2 * idx + 1](x)

            x = self.linears[-1](x)

            return x

    def diffusion_loss_fn(
        model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps
    ):
        """Calculate loss by sampling at any time t"""
        batch_size = x_0.shape[0]

        t = torch.randint(0, n_steps, size=(batch_size // 2,))
        t = torch.cat([t, n_steps - 1 - t], dim=0)
        t = t.unsqueeze(-1)

        a = alphas_bar_sqrt[t]

        am1 = one_minus_alphas_bar_sqrt[t]

        e = torch.randn_like(x_0)

        x = x_0 * a + e * am1

        output = model(x, t.squeeze(-1).to(device))

        return (e - output).square().mean()

    def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
        """Infer x[T-1], x[T-2], ..., x[0] from x[T]"""

        cur_x = torch.randn(shape).to(device)
        x_seq = [cur_x]
        for i in reversed(range(n_steps)):
            cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
            x_seq.append(cur_x)
        return x_seq

    def guided_p_sample_loop(
        model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, coef_lcb=0.1
    ):
        """Infer x[T-1], x[T-2], ..., x[0] from x[T] with guidance"""
        torch.set_grad_enabled(True)

        cur_x = torch.randn(shape, requires_grad=True, device=device)
        x_seq = [cur_x]
        for i in reversed(range(n_steps)):
            cur_x_tmp = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)

            cur_x_tmp_np = cur_x_tmp.detach().cpu().numpy()

            eval_result = surrogate_model.evaluate(
                cur_x_tmp_np, std=True, calc_gradient=True
            )
            mean = torch.from_numpy(eval_result["F"]).to(device)
            mean_grad = torch.from_numpy(eval_result["dF"]).to(device)
            std = torch.from_numpy(eval_result["S"]).to(device)
            std_grad = torch.from_numpy(eval_result["dS"]).to(device)

            value = mean - coef_lcb * std
            value_grad = mean_grad - coef_lcb * std_grad

            w_j = calculate_entropy_weights(value)

            weighted_value_grad = torch.zeros_like(value_grad[:, 0, :])
            for i in range(value_grad.shape[1]):
                weighted_value_grad += value_grad[:, i, :] * w_j[i]

            weighted_value_grad = nn.functional.normalize(
                weighted_value_grad, p=2, dim=-1
            )

            cur_x = guided_p_sample(
                model,
                cur_x,
                i,
                betas,
                one_minus_alphas_bar_sqrt,
                weighted_value_grad,
                grad_scale=0.1,
            )

            cur_x = cur_x.detach().requires_grad_(True)
            x_seq.append(cur_x)

        return x_seq

    def calculate_entropy_weights(value):

        min_val = value.min(0, keepdim=True)[0]
        max_val = value.max(0, keepdim=True)[0]
        normalized_value = (value - min_val) / (max_val - min_val + 1e-10)

        sum_normalized = normalized_value.sum(0, keepdim=True)
        P_ij = normalized_value / (sum_normalized + 1e-10)

        epsilon = 1e-10
        E_j = -(
            1 / torch.log(torch.tensor(P_ij.shape[0], dtype=torch.float))
        ) * torch.sum(P_ij * torch.log(P_ij + epsilon), dim=0)

        w_j = (1 - E_j) / (1 - E_j).sum()

        return w_j

    def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
        """Sample reconstructed value at time t from x[T]"""

        t = torch.tensor([t]).to(device)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

        eps_theta = model(x, t)

        mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

        z = torch.randn_like(x)
        sigma_t = betas[t].sqrt()

        sample = mean + nonzero_mask * sigma_t * z

        return sample

    def guided_p_sample(
        model, x, t, betas, one_minus_alphas_bar_sqrt, grad, grad_scale=0.1
    ):
        """Guided sampling of reconstructed value at time t from x[T]"""
        t = torch.tensor([t], device=device)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

        eps_theta = model(x, t)

        mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

        if grad is not None:
            grad = grad.to(mean.dtype)
            variance = one_minus_alphas_bar_sqrt[t] ** 2
            new_mean = mean + variance * grad

        z = torch.randn_like(x)
        sigma_t = betas[t].sqrt()

        if grad is not None:
            sample = new_mean + nonzero_mask * sigma_t * z

        return sample

    seed = 3407

    batch_size = 4096
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    num_epoch = 4000

    model = MLPDiffusion(num_steps)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(num_epoch):
        for idx, batch_x in enumerate(dataloader):
            loss = diffusion_loss_fn(
                model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    new_offsprings = []

    for i in range(100):
        x_seq = guided_p_sample_loop(
            model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt
        )
        new_x = x_seq[-1].detach()
        new_x = new_x.cpu().numpy()
        new_offsprings.append(new_x)

    for i in range(1000):
        x_seq = p_sample_loop(
            model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt
        )
        new_x = x_seq[-1].detach()
        new_x = new_x.cpu().numpy()
        new_offsprings.append(new_x)

    result = np.vstack(new_offsprings)

    return result
