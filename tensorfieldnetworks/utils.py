
import torch
import numpy as np
import scipy.linalg

from torch.nn import functional

FLOAT_TYPE = torch.float32
EPSILON = torch.Tensor([1e-8])


def difference_matrix(geometry):
    """
    Get relative vector matrix for array of shape [N, 3].

    Args:
        geometry: tf.Tensor with Cartesian coordinates and shape [N, 3]

    Returns:
        Relative vector matrix with shape [N, N, 3]
    """
    # [N, 1, 3]
    ri = geometry.unsqueeze(-2)
    # [1, N, 3]
    rj = geometry.unsqueeze(-3)
    # [N, N, 3]
    rij = ri - rj
    return rij

def get_eijk():
    """
    Constant Levi-Civita tensor

    Returns:
        tf.Tensor of shape [3, 3, 3]
    """
    eijk_ = np.zeros((3, 3, 3))
    eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
    eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
    return torch.Tensor(eijk_).to(FLOAT_TYPE)


def norm_with_epsilon(input_tensor, axis=None, keep_dims=False):
    """
    Regularized norm

    Args:
        input_tensor: tf.Tensor

    Returns:
        tf.Tensor normed over axis
    """
    return torch.sqrt(torch.max(torch.sum(input_tensor**2, dim=axis, keepdim=keep_dims), EPSILON))

def ssp(x):
        return torch.log(0.5 * torch.exp(x) + 0.5 )


def rotation_equivariant_nonlinearity(x, nonlin=ssp, biases=None):
    """
    Rotation equivariant nonlinearity.

    The -1 axis is assumed to be M index (of which there are 2 L + 1 for given L).

    Args:
        x: tf.Tensor with channels as -2 axis and M as -1 axis.

    Returns:
        tf.Tensor of same shape as x with 3d rotation-equivariant nonlinearity applied.
    """
    shape = x.size()
    representation_index = shape[-1]
    if representation_index == 1:
        return nonlin(x)
    else:
        norm = norm_with_epsilon(x, axis=-1)
        nonlin_out = nonlin(norm + biases)
        factor = nonlin_out/norm
        # Expand dims for representation index.
        return x * factor.unsqueeze(-1)


def distance_matrix(geometry):
    """
    Get relative distance matrix for array of shape [N, 3].

    Args:
        geometry: tf.Tensor with Cartesian coordinates and shape [N, 3]

    Returns:
        Relative distance matrix with shape [N, N]
    """
    # [N, N, 3]
    rij = difference_matrix(geometry)
    # [N, N]
    dij = norm_with_epsilon(rij, axis=-1)
    return dij

def random_rotation_matrix(numpy_random_state):
    """
    Generates a random 3D rotation matrix from axis and angle.

    Args:
        numpy_random_state: numpy random state object

    Returns:
        Random rotation matrix.
    """
    rng = numpy_random_state
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis) + 1e-8
    theta = 2 * np.pi * rng.uniform(0.0, 1.0)
    return rotation_matrix(axis, theta)

def rotation_matrix(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis * theta))
