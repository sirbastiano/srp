import torch
import torch.nn.functional as F
import math

def loss_split_mse(pred: torch.Tensor, target: torch.Tensor):
    # pred, target: complex tensors (dtype=torch.cfloat)
    return F.mse_loss(pred.real, target.real) + F.mse_loss(pred.imag, target.imag)

def loss_complex_l2(pred: torch.Tensor, target: torch.Tensor):
    diff = pred - target
    return torch.mean(torch.abs(diff).pow(2))

def loss_complex_l1(pred: torch.Tensor, target: torch.Tensor):
    diff = pred - target
    return torch.mean(torch.abs(diff))

def loss_mag_l2(pred: torch.Tensor, target: torch.Tensor):
    return torch.mean((pred.abs() - target.abs()).pow(2))

def loss_log_mag(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    # log magnitude error (for dynamic range)
    return torch.mean((torch.log(pred.abs() + eps) - torch.log(target.abs() + eps)).pow(2))

def loss_amp_phase(pred: torch.Tensor, target: torch.Tensor, w_phase=1.0):
    mag_err = torch.mean((pred.abs() - target.abs()).pow(2))
    # relative phase: angle(pred * conj(target))
    phase_err = torch.mean(torch.angle(pred * target.conj()).pow(2))
    return mag_err + w_phase * phase_err

def loss_power_consistency(pred: torch.Tensor, target: torch.Tensor):
    p_pred = torch.sum(pred.abs().pow(2))
    p_tgt = torch.sum(target.abs().pow(2))
    return (p_pred - p_tgt).pow(2)

def loss_symmetry(pred: torch.Tensor, sym_op):
    # sym_op: function or lambda implementing symmetry (e.g. mirror, conj)
    # returns same shape as pred
    return torch.mean(torch.abs(pred - sym_op(pred)).pow(2))

def loss_complex_tv(pred: torch.Tensor, p=1):
    # finite differences in spatial dimensions (assuming shape: B × C × H × W or B×H×W)
    # generalize to correct dims
    # here assume last two dims are spatial
    dx = pred[..., 1:, :] - pred[..., :-1, :]
    dy = pred[..., :, 1:] - pred[..., :, :-1]
    return (torch.abs(dx).pow(p).mean() + torch.abs(dy).pow(p).mean())

def loss_phase_smooth(pred: torch.Tensor):
    def ph_diff(a, b):
        return torch.angle(a * b.conj())
    dx = ph_diff(pred[..., 1:, :], pred[..., :-1, :])
    dy = ph_diff(pred[..., :, 1:], pred[..., :, :-1])
    return (dx.pow(2).mean() + dy.pow(2).mean())

def loss_grad_match(pred: torch.Tensor, target: torch.Tensor):
    # match gradients in magnitude + phase
    # magnitude gradients
    gm_x = (pred.abs()[..., 1:, :] - pred.abs()[..., :-1, :])
    gm_y = (pred.abs()[..., :, 1:] - pred.abs()[..., :, :-1])
    gt_x = (target.abs()[..., 1:, :] - target.abs()[..., :-1, :])
    gt_y = (target.abs()[..., :, 1:] - target.abs()[..., :, :-1])
    mag_grad_err = (gm_x - gt_x).pow(2).mean() + (gm_y - gt_y).pow(2).mean()
    # phase gradients (relative)
    def phdiff(a, b):
        return torch.angle(a * b.conj())
    pdx = phdiff(pred[..., 1:, :], pred[..., :-1, :])
    pdy = phdiff(pred[..., :, 1:], pred[..., :, :-1])
    tdx = phdiff(target[..., 1:, :], target[..., :-1, :])
    tdy = phdiff(target[..., :, 1:], target[..., :, :-1])
    phase_grad_err = (pdx - tdx).pow(2).mean() + (pdy - tdy).pow(2).mean()
    return mag_grad_err + phase_grad_err

def loss_speckle_nll(pred: torch.Tensor, target: torch.Tensor, variance: float = 1.0):
    # negative log‐likelihood under simple complex Gaussian noise model:
    # p(target | pred) ∝ exp( - |target - pred|^2 / variance )
    diff = target - pred
    # ignoring constants:
    return torch.mean(torch.abs(diff).pow(2)) / variance

def loss_adversarial(pred: torch.Tensor, discriminator, real_label=1.0, fake_label=0.0):
    # discriminator expects complex input (or split) and returns logits or prob
    # For generator loss: wants discriminator(pred) => real
    out = discriminator(pred)
    loss = F.binary_cross_entropy_with_logits(out, torch.full_like(out, real_label))
    return loss

def loss_feature(pred: torch.Tensor, target: torch.Tensor, feature_extractor):
    # feature_extractor: function mapping complex -> feature tensor (real domain or complex)
    return F.mse_loss(feature_extractor(pred), feature_extractor(target))

# Example of composite / weighted loss
class ComplexLossComposite:
    def __init__(self, weights: dict, sym_op=None, feature_extractor=None, discriminator=None):
        """
        weights: dict mapping loss name to weight, e.g. {'l2':1.0, 'tv':0.1, 'power':0.01}
        sym_op: optional symmetry operator function
        feature_extractor: optional
        discriminator: optional
        """
        self.weights = weights
        self.sym_op = sym_op
        self.feature_extractor = feature_extractor
        self.discriminator = discriminator

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, **kwargs):
        total = 0.0
        w = self.weights
        if w.get('split_mse', 0) > 0:
            total = total + w['split_mse'] * loss_split_mse(pred, target)
        if w.get('l2', 0) > 0:
            total = total + w['l2'] * loss_complex_l2(pred, target)
        if w.get('l1', 0) > 0:
            total = total + w['l1'] * loss_complex_l1(pred, target)
        if w.get('mag_l2', 0) > 0:
            total = total + w['mag_l2'] * loss_mag_l2(pred, target)
        if w.get('log_mag', 0) > 0:
            total = total + w['log_mag'] * loss_log_mag(pred, target)
        if w.get('amp_phase', 0) > 0:
            total = total + w['amp_phase'] * loss_amp_phase(pred, target)
        if w.get('power', 0) > 0:
            total = total + w['power'] * loss_power_consistency(pred, target)
        if w.get('tv', 0) > 0:
            total = total + w['tv'] * loss_complex_tv(pred)
        if w.get('ph_smooth', 0) > 0:
            total = total + w['ph_smooth'] * loss_phase_smooth(pred)
        if w.get('grad_match', 0) > 0:
            total = total + w['grad_match'] * loss_grad_match(pred, target)
        if w.get('speckle', 0) > 0:
            total = total + w['speckle'] * loss_speckle_nll(pred, target, kwargs.get('variance', 1.0))
        if w.get('sym', 0) > 0 and self.sym_op is not None:
            total = total + w['sym'] * loss_symmetry(pred, self.sym_op)
        if w.get('feat', 0) > 0 and self.feature_extractor is not None:
            total = total + w['feat'] * loss_feature(pred, target, self.feature_extractor)
        if w.get('adv', 0) > 0 and self.discriminator is not None:
            total = total + w['adv'] * loss_adversarial(pred, self.discriminator)
        return total