"""

References:
- https://github.com/mmatena/model_merging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..utils import get_mergeable_variables


def compute_fisher_matrices(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
) -> list[torch.Tensor]:
    model.eval()
    variables = get_mergeable_variables(model)

    fishers = [torch.zeros_like(w, requires_grad=False) for w in variables]
    n_examples = 0

    for batch in tqdm(dataloader, desc="computing fisher matrix"):
        batch = {
            k: v.to(next(model.parameters()).device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        batch_size = batch["input_ids"].shape[0]
        n_examples += batch_size
        num_labels = model.config.num_labels
        batch_fishers = []

        for i in range(batch_size):
            single_example = {k: v[i : i + 1] for k, v in batch.items()}

            model.zero_grad()
            with torch.no_grad():
                outputs = model(**single_example)
                logits = outputs.logits.squeeze(0)
                log_probs = F.log_softmax(logits, dim=-1)
                probs = F.softmax(logits, dim=-1)

            example_fishers = []
            for label_idx in range(num_labels):
                model.zero_grad()
                log_prob = log_probs[label_idx].requires_grad_(True)
                log_prob.backward(retain_graph=True)

                sq_grads = []
                for param in variables:
                    if param.grad is not None:
                        sq_grad = probs[label_idx] * param.grad.detach().pow(2)
                        sq_grads.append(sq_grad)
                    else:
                        sq_grads.append(torch.zeros_like(param))

                example_fishers.append(sq_grads)

            summed_fishers = []
            for param_idx in range(len(variables)):
                param_fisher = sum(
                    example_fishers[class_idx][param_idx]
                    for class_idx in range(num_labels)
                )
                summed_fishers.append(param_fisher)

            if not batch_fishers:
                batch_fishers = summed_fishers
            else:
                batch_fishers = [
                    bf + sf for bf, sf in zip(batch_fishers, summed_fishers)
                ]

        for f_idx, (fisher, batch_fisher) in enumerate(zip(fishers, batch_fishers)):
            fishers[f_idx] = fisher + batch_fisher

    fishers = [fisher / n_examples for fisher in fishers]
    return fishers


def fisher_merge(
    params: list[list[torch.nn.Parameter]],
    merged_params: list[torch.nn.Parameter],
    coeff: list,
    aux_matrices: list,
):
    assert len({len(merged_params)} | set(len(v) for v in params)) == 1
    assert len(aux_matrices) == len(params)

    for idx, merged_param in enumerate(merged_params):
        lhs, rhs = [], []
        for i, (param, coefficient, aux) in enumerate(zip(params, coeff, aux_matrices)):
            diag = aux if isinstance(aux, float) else aux[idx]
            if isinstance(diag, float):
                diag = torch.full_like(param[idx], diag)
            if i == 0:
                diag = torch.maximum(torch.full_like(diag, 1e-6), diag)
            lhs.append(coefficient * diag)
            rhs.append(coefficient * diag * param[idx])

        rhs_sum = torch.sum(torch.stack(rhs), dim=0)
        lhs_sum = torch.sum(torch.stack(lhs), dim=0)
        merged_param.data.copy_(rhs_sum / lhs_sum)

    return merged_params
