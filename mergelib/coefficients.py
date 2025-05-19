import torch


def random_coefficients(num_models: int, num_coefficients: int) -> list[float]:
    distribution = torch.distributions.dirichlet.Dirichlet(torch.ones(num_models))
    return distribution.sample(sample_shape=torch.Size([num_coefficients])).tolist()


def grid_coefficients(num_coefficients: int) -> list[float]:
    num_coefficients -= 2
    denominator = num_coefficients + 1
    coefficients = [
        ((i + 1) / denominator, 1 - (i + 1) / denominator)
        for i in range(num_coefficients)
    ]
    coefficients = [(0.0, 1.0)] + coefficients + [(1.0, 0.0)]
    coefficients.reverse()
    return coefficients
