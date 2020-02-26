import torch
import torch.nn.functional as F
import torch.distributions as distributions


def split_mask(n_features):
    """
    Create a simple binary mask for splitting the features in half.
    """
    mask = torch.zeros(n_features, dtype=int)

    mask[n_features//2:] = 1

    return mask.unsqueeze(0)


def dequantize(x, constraint=0.9, inverse=False):
    if inverse:
        x = 2 / (torch.exp(-x) + 1) - 1
        x /= constraint
        x = (x + 1) / 2

        return x, 0
    else:
        B, C, H, W = x.shape

        noise = distributions.Uniform(0., 1.).sample(x.shape)
        x = (x * 255 + noise) / 256

        return x

        x = x * 2 - 1
        x *= constraint
        x = (x + 1) / 2

        logit_x = x.log() - (1 - x).log()
        pre_logit_scale = torch.Tensor([constraint]).log() \
            - torch.Tensor([1 - constraint]).log()
        log_det_J = F.softplus(logit_x) + F.softplus(-logit_x) \
            - F.softplus(-pre_logit_scale)

        return logit_x, log_det_J.view(B, -1).sum(1, keepdim=True)


def load_model(path, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    return model.eval()
