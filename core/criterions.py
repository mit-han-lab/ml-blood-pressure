import torch
import torch.nn.functional as F


def complex_mse(output, target):
    return F.mse_loss(torch.view_as_real(output),
                      torch.view_as_real(target))

def std_error_over_mean_error(output, target):
    error = target - output
    return torch.std(error) / (torch.mean(error) + 1e-5)

def std_error_over_mean_error_abs(output, target):
    error = (target - output).abs()
    return torch.std(error) / (torch.mean(error) + 1e-5)

def std_error_times_mean_error_abs(output, target):
    error = (target - output).abs()
    return torch.std(error) * (torch.mean(error) + 1e-5)

    # return (torch.view_as_real(output)
    #         - torch.view_as_real(target)).abs().mean()

def std_error(output, target):
    error = target - output
    return torch.std(error)

