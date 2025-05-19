from typing import Optional

import torch
from evaluate import EvaluationModule
from tqdm import tqdm

from .coefficients import grid_coefficients, random_coefficients
from .engine.fisher import compute_fisher_matrices, fisher_merge
from .utils import clone, get_mergeable_variables, set_mergeable_variables


def merge(
    models: list,
    dataset,
    metric: EvaluationModule,
    coefficient_type: Optional[str] = "grid",
    method: Optional[str] = "fisher",
    num_coefficients: Optional[int] = 51,
    normalize: Optional[bool] = True,
):
    # ======== create new model ========
    merged_model = clone(models[0])
    merged_params = get_mergeable_variables(merged_model)
    params = [get_mergeable_variables(model) for model in models]
    assert len({len(merged_params)} | set(len(v) for v in params)) == 1

    # ======== create coefficients ========
    if coefficient_type == "grid":
        assert len(models) == 2
        coefficients = grid_coefficients(num_coefficients=num_coefficients)
    elif coefficient_type == "random":
        coefficients = random_coefficients(
            num_models=len(models), num_coefficients=num_coefficients
        )
    else:
        raise ValueError(f"Invalid coefficient type: {coefficient_type}")

    # ======== compute auxilary matrices ========
    if method == "fisher":
        aux_matrices = [compute_fisher_matrices(model, dataset) for model in models]
    elif method == "isotropic":
        aux_matrices = [1.0] * len(models)
    else:
        raise ValueError(f"Invalid method: {method}")

    assert len(aux_matrices) == len(models)

    if normalize:
        for aux_matrix in aux_matrices:
            norm = torch.sqrt(sum(torch.sum(d**2) for d in aux_matrix))
            aux_matrix = [d / norm for d in aux_matrix]

    # ======== merge and evaluate for each coefficient ========
    results_and_params = []
    for coeff in tqdm(coefficients, desc="trying different coefficients"):
        fisher_merge(
            params=params,
            merged_params=merged_params,
            coeff=coeff,
            aux_matrices=aux_matrices,
        )

        # ======== evaluate ========
        for batch in dataset:
            outputs = merged_model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            y = batch["labels"]
            metric.add_batch(predictions=preds, references=y)
        result = metric.compute()

        current_params = [p.clone().detach() for p in merged_params]
        results_and_params.append((result, current_params, coeff))

    # TODO(@SauravMaheshkar): probably a better way to do this
    def get_score(res):
        if isinstance(res, dict):
            return list(res.values())[0]
        return res

    best_result, best_params, best_coeff = max(
        results_and_params, key=lambda x: get_score(x[0])
    )

    set_mergeable_variables(merged_model, best_params)

    return merged_model, best_result, best_coeff
