import numpy as np

def relu(x: np.ndarray)->np.ndarray:
    """
    Applied ReLU element-wise
    """
    return np.maximum(0, x)

def conv2d(x: np.ndarray, weights: np.ndarray, bias: np.ndarray, padding:int=1)->np.ndarray:
    N, H, W, C_in = x.shape
    K = weights.shape[0]
    C_out = weights.shape[-1]

    x_padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="constant")
    out = np.zeros((N, H, W, C_out))

    for i in range(H):
        for j in range(W):
            patch = x_padded[:, i:i+K, j:j+K, :]
            out[:, i, j, :] = np.einsum("nklc,klco->no", patch, weights)+bias

    return out
    
def vgg_conv_block(x: np.ndarray, num_convs: int, out_channels: int) -> np.ndarray:
    """
    Implement a VGG-style convolutional block.
    """
    out = x
    kernel_size = 3

    for step in range(num_convs):
        current_in_channels = out.shape[-1]
        W = np.random.randn(kernel_size, kernel_size, current_in_channels, out_channels) * 0.01
        b = np.zeros(out_channels)

        out = conv2d(out, W, b, padding=1)
        out = relu(out)
    return out

