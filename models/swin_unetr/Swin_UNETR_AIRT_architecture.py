# Standard library imports
import itertools
from collections.abc import Sequence
from typing import Final

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing import Type
from einops import rearrange

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: Number of input feature channels.
            num_heads: Number of attention heads.
            window_size: Size of the local window for attention.
            qkv_bias: If True, adds a learnable bias to query, key, value projections.
            attn_drop: Dropout rate for attention weights.
            proj_drop: Dropout rate for output projection.
        """

        super().__init__()
        self.dim = dim  # Dimension of input features
        self.window_size = window_size  # Size of the attention window
        self.num_heads = num_heads  # Number of attention heads

        # Dimension per attention head
        head_dim = dim // num_heads
        # Scaling factor for attention scores to prevent large values during softmax
        self.scale = head_dim**-0.5
        # Check for meshgrid arguments compatibility
        mesh_args = torch.meshgrid.__kwdefaults__

        # Handle 3D window sizes (e.g., for 3D volumes)
        if len(self.window_size) == 3:
            # Create a parameter table for relative position biases
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            # Create coordinate grids for relative position computation
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            # Generate coordinate grids with indexing support
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)  # Flatten the coordinates for easier computation
            # Compute relative coordinates between each point
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Permute dimensions for indexing
            # Adjust relative coordinates for bias table indexing
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            # Map to flattened indices
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        else: # Handle other input tensors
            raise ValueError("Unsupported dimensions. Expected input to have length of 3 dimensions.")

        # Register the relative position index as a buffer
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # Define linear layers for query, key, value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Dropout layers for attention and output projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Initialize relative position bias table with truncated normal distribution
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        # Softmax layer for attention normalization
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        Forward pass for window-based self-attention.

        Args:
            x: Input tensor of shape (num_windows, num_tokens, embed_dim).
            mask: Attention mask to restrict certain positions.

        Returns:
            Tensor of shape (batch_size, num_tokens, embed_dim) after self-attention.
        """
        b, n, c = x.shape
        # Compute query, key, and value projections and reshape them
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Separate into query, key, and value tensors
        q = q * self.scale  # Scale query for better numerical stability
        attn = q @ k.transpose(-2, -1)  # Compute dot-product attention scores

        # Add relative position bias to attention scores
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)  # Broadcast bias across batch and heads

        # Apply attention mask if provided
        if mask is not None:
            nw = mask.shape[0]  # Number of windows
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)  # Apply softmax to normalize attention scores
        else:
            attn = self.softmax(attn)  # Apply softmax to normalize attention scores

        # Apply dropout to attention weights
        attn = self.attn_drop(attn).to(v.dtype)
        # Compute attention-weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        # Apply linear projection and dropout to the output
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """Window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    This function partitions an input tensor into smaller windows based on the specified window size.
    This is used in Swin Transformer models to divide the input into regions for applying self-attention efficiently.

    Args:
        x (Tensor): Input tensor of shape (batch_size, depth, height, width, channels) for 3D input data.
        window_size (Sequence[int]): The size of each local window for partitioning.
                                     It should be a tuple specifying the size for each spatial dimension (depth, height, width).

    Returns:
        Tensor: Partitioned tensor of shape (num_windows, window_size_product, channels) where `num_windows`
                is the total number of windows, and `window_size_product` is the product of the window dimensions.

    Example:
        >>> import torch
        >>> # Example 3D tensor with batch size 1, depth 8, height 8, width 8, and 3 channels
        >>> x = torch.arange(1, 1 * 8 * 8 * 8 * 3 + 1).view(1, 8, 8, 8, 3)
        >>> window_size = (4, 4, 4)
        >>> windows = window_partition(x, window_size)
        >>> print("Shape of partitioned windows:", windows.shape)
        >>> # Output shape: (8, 4*4*4, 3) -> (num_windows, window_size_product, channels)
    """

    # Get the shape of the input tensor
    x_shape = x.size()

    # Handle 5D input tensors (e.g., 3D input data with channels and batch dimensions)
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape  # Extract batch, depth, height, width, and channels

        # Reshape the input tensor so that each dimension becomes divisible by the window size
        # This effectively creates smaller windows within the tensor
        x = x.view(
            b,
            d // window_size[0],  # Number of windows along the depth dimension
            window_size[0],       # Size of each window along the depth dimension
            h // window_size[1],  # Number of windows along the height dimension
            window_size[1],       # Size of each window along the height dimension
            w // window_size[2],  # Number of windows along the width dimension
            window_size[2],       # Size of each window along the width dimension
            c                     # Channels (kept the same)
        )

        # Rearrange the dimensions to bring window dimensions next to each other and flatten each window
        # Permute moves the dimensions around to the specified order, making it ready for further processing
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7)  # Change order of dimensions for easier window processing
             .contiguous()                     # Ensures that data is stored contiguously in memory
             .view(-1, window_size[0] * window_size[1] * window_size[2], c)  # Flatten each window
        )
    else:
        raise ValueError("Unsupported dimensions. Expected input to have length of 5 dimensions (b, d, h, w, c).")

    return windows


def window_reverse(windows, window_size, dims):
    """
    Window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    <https://arxiv.org/abs/2103.14030>
    https://github.com/microsoft/Swin-Transformer

    This function reverses the window partitioning process and reconstructs the original spatial dimensions
    from the smaller windows. It reassembles the partitioned windows back into their original spatial arrangement.

    Args:
        windows: Tensor representing partitioned windows. Shape typically is (num_windows, window_size_product, channels).
        window_size: Size of the local window (e.g., (depth, height, width) for 3D).
        dims: Dimension values of the original spatial dimensions (before window partitioning).

    Returns:
        Tensor of shape corresponding to the original dimensions before window partitioning.

    Example:
        >>> import torch
        >>> # Example 3D tensor with batch size 1, depth 8, height 8, width 8, and 3 channels (partitioned into windows)
        >>> windows = torch.randn(8, 4*4*4, 3)  # 8 windows, each with size 4x4x4 and 3 channels
        >>> window_size = (4, 4, 4)
        >>> dims = (1, 8, 8, 8)  # Original dimensions: (batch_size, depth, height, width)
        >>> x = window_reverse(windows, window_size, dims)
        >>> print("Shape of reconstructed tensor:", x.shape)
        >>> # Output shape: (1, 8, 8, 8, 3)
    """

    # Handle 3D data (e.g., 5D input tensor with shape (batch_size, depth, height, width, channels))
    if len(dims) == 4:
        b, d, h, w = dims  # Unpack batch size and spatial dimensions (depth, height, width)

        # Reshape windows to form a structured tensor with individual window dimensions reassembled
        x = windows.view(
            b,
            d // window_size[0],  # Number of windows along the depth dimension
            h // window_size[1],  # Number of windows along the height dimension
            w // window_size[2],  # Number of windows along the width dimension
            window_size[0],  # Depth of each window
            window_size[1],  # Height of each window
            window_size[2],  # Width of each window
            -1,  # Number of channels (kept the same)
        )

        # Permute dimensions to restore the original spatial arrangement by rearranging window dimensions
        # The order of permutation restores the windows to their original tensor layout
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    else:
        # Raise an error if unsupported dimensions are provided
        raise ValueError("Unsupported dimensions. Expected input to have length of 4 dimensions (b, d, h, w).")

    # Return the tensor with original spatial dimensions restored
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Computing window size and optional shift size adjustments based on the input size.

    This function adjusts the window size and shift size based on the dimensions of the input (`x_size`).
    If the input size for a specific dimension is smaller than or equal to the corresponding window size,
    the function sets the window size to the input size and the shift size to zero for that dimension.

    Args:
        x_size (tuple): The input size as a tuple of dimensions (e.g., height, width, depth).
        window_size (tuple): The local window size for each dimension.
        shift_size (tuple, optional): The amount to shift the window. Defaults to None.

    Returns:
        tuple: Adjusted window size. If `shift_size` is provided, also returns the adjusted shift size.

    Example:
        >>> x_size = (10, 20, 15)
        >>> window_size = (7, 7, 7)
        >>> shift_size = (3, 3, 3)
        >>> get_window_size(x_size, window_size, shift_size)
        ((7, 7, 7), (0, 3, 3))
    """
    # Create a mutable list from the provided window_size for adjustments
    use_window_size = list(window_size)
    if shift_size is not None:
        # Create a mutable list from the provided shift_size for adjustments
        use_shift_size = list(shift_size)

    # Iterate over each dimension of the input size
    for i in range(len(x_size)):
        # If the input size in the current dimension is less than or equal to the window size
        if x_size[i] <= window_size[i]:
            # Adjust the window size to match the input size
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                # Set the shift size to 0 for this dimension since the window size matches input size
                use_shift_size[i] = 0

    # Return adjusted window size and optionally adjusted shift size
    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def compute_mask(dims, window_size, shift_size, device):
    """
    Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-TransformerComputing region masks based on: "Liu et al., Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    https://arxiv.org/abs/2103.14030

    This function divides the input tensor into regions, assigns a unique integer label to each region, and creates an
    attention mask (`attn_mask`) by comparing labels within and across windows. This mask restricts attention computations
    to only valid elements, optimizing efficiency and adhering to the Swin Transformer's local attention mechanism.

    Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        # For a 3D input tensor (depth, height, width), create an initial mask filled with zeros
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)

        # Dividing the input tensor into 3D regions using slices for depth, height, and width dimensions
        # Each dimension is divided into three slices based on the window and shift sizes.
        # Example window_size = (7, 7, 7), shift_size = (3, 3, 3) will divide each dimension as:
        # - slice(-window_size[0]):  Covers elements from the beginning up to index, in this case, -7 (not inclusive)
        # - slice(-window_size[0], -shift_size[0]): Covers elements between indices `-7` to `-3`
        # - slice(-shift_size[0], None): Covers the last `3` elements
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    # Assign a unique integer label to each region within the 3D space
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
    elif len(dims) == 2:
        raise ValueError("2D input is not supported. Please provide a 3D input with dimensions (d, h, w).")
    else:
        raise ValueError("Unsupported dimensions. Expected input to have length of 3 dimensions (d, h, w).")

    # The `img_mask` tensor, which contains unique integer labels for different regions of the input tensor,
    # is now partitioned into smaller windows of size specified by `window_size` using the `window_partition` function.
    # This function divides the tensor spatially into separate non-overlapping windows for localized processing,
    # facilitating efficient computation of self-attention in the Swin Transformer by focusing within each window.
    # mask_windows => tensor of shape (num_windows, window_size_product, 1)
    mask_windows = window_partition(img_mask, window_size)

    # Since `img_mask` initially had an extra singleton dimension (i.e., shape (1, d, h, w, 1) for 3D data),
    # we remove this last dimension using `squeeze(-1)`.
    # This operation reduces the dimensionality of `mask_windows` by eliminating the singleton dimension,
    # resulting in a tensor that contains the labels of regions in each window.
    # mask_windows => tensor of shape (num_windows, window_size_product)
    mask_windows = mask_windows.squeeze(-1)

    # Create an attention mask for controlling the attention mechanism in the Swin Transformer.
    # The attention mask is generated by comparing the labels of elements in different windows:
    # - `mask_windows.unsqueeze(1)` expands the dimensions of `mask_windows` for broadcasting so that
    #   each window label can be compared with every other window label.
    # - `mask_windows.unsqueeze(2)` similarly expands the dimensions of `mask_windows` for element-wise comparisons.

    # The subtraction operation (`mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)`) generates a tensor
    # that contains zero values when elements belong to the same window and non-zero values otherwise.
    # This tensor effectively encodes information about which elements can attend to each other:
    # - Elements with zero values (same window labels) can attend to each other.
    # - Elements with non-zero values (different window labels) cannot attend to each other.

    # The mask is further refined using `masked_fill`:
    # - `masked_fill(attn_mask != 0, float(-100.0))` sets large negative values (-100.0) for elements
    #   that belong to different windows, effectively blocking attention between them by making their
    #   attention scores very low (close to negative infinity).
    # - `masked_fill(attn_mask == 0, float(0.0))` sets zero values for elements within the same window,
    #   allowing attention between them without modification of their scores.
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


    return attn_mask


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: Number of input feature channels.
            num_heads: Number of attention heads in the multi-head self-attention mechanism.
            window_size: Size of the local window for attention computations.
            shift_size: Size of the shift applied to the window during the shifted-window mechanism.
            mlp_ratio: Ratio of the hidden dimension size in the MLP to the embedding dimension size.
            qkv_bias: Boolean indicating whether to add a bias term to the query, key, and value tensors.
            drop: Dropout rate for the final output projection.
            attn_drop: Dropout rate for the attention scores.
            drop_path: Drop path (stochastic depth) rate.
            act_layer: Activation function used in the MLP layers (e.g., GELU).
            norm_layer: Normalization layer applied before and after attention (default: LayerNorm).
            use_checkpoint: If True, use gradient checkpointing to save memory during training.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # Layer normalization before self-attention
        self.norm1 = norm_layer(dim)

        # Window-based multi-head self-attention mechanism
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Optional drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Second normalization layer
        self.norm2 = norm_layer(dim)

        # MLP block with one hidden layer
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        """
        Applies the first part of the forward pass in the Swin Transformer block, including:
          - 1. Layer normalization
          - 2. Handling input shape and padding
          - 3. Applying window partitioning and shifted window self-attention
          - 4. Reversing window operations
          - 5. Removing padding (if applied)

        Args:
            x (Tensor): Input tensor of shape (batch_size, depth, height, width, channels) or 4D equivalent.
            mask_matrix (Tensor): Precomputed attention mask to control self-attention behavior.

        Returns:
            Tensor: Output tensor after applying all transformations.
        """
        # Get the shape of the input tensor (could be 5D or 4D)
        x_shape = x.size()

        # Apply layer normalization to stabilize and optimize the learning process
        x = self.norm1(x)

        # Handle 5D input tensors (e.g., 3D images)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape  # Unpack the dimensions: batch size, depth, height, width, and channels

            # Calculate effective window size and shift size based on input dimensions
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)

            # Calculate padding needed to make dimensions divisible by the window size
            # No padding on the left/top/front sides (pad_l, pad_t, pad_d0 are zero)
            pad_l = pad_t = pad_d0 = 0
            # Calculate padding for the right/bottom/back sides to ensure divisibility
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]  # Depth dimension padding
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]  # Height dimension padding
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]  # Width dimension padding

            # Apply padding to the input tensor to ensure its dimensions are divisible by the window size
            # Padding order: (width padding right, width padding left, height padding bottom, height padding top, ...)
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))

            # Update dimension variables to reflect changes after padding
            _, dp, hp, wp, _ = x.shape  # dp, hp, wp are the new depth, height, and width after padding
            dims = [b, dp, hp, wp]  # Store updated dimensions

        else:  # Handle other input tensors (e.g., 2D input data with channels and batch dimensions)
            raise ValueError("Unsupported dimensions. Expected input to have length of 5 dimensions (b, d, h, w, c).")

        # Check if any shift is required (shift_size > 0)
        if any(i > 0 for i in shift_size):
            # Apply negative rolling shift along depth, height, and width if input is 5D
            if len(x_shape) == 5:
                # Roll (shift) elements along depth, height, and width dimensions (CYCLE SHIFT)
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            # Set attention mask to the precomputed mask matrix for shifted attention
            attn_mask = mask_matrix
        else:
            # No shift is needed, retain original input
            shifted_x = x
            attn_mask = None  # No attention mask is required for non-shifted windows

        # Partition the (shifted) input tensor into windows for applying window-based self-attention
        x_windows = window_partition(shifted_x, window_size)
        # After window partitioning:
        # - For 3D input, x_windows has shape (num_windows, window_d * window_h * window_w, channels)
        # Here, `num_windows` is the number of windows formed, `window_d`, `window_h`, `window_w` are
        # window dimensions, and `channels` is the number of feature channels.

        # Apply window-based self-attention mechanism using `self.attn`
        # This computes self-attention independently within each window
        attn_windows = self.attn(x_windows, mask=attn_mask)
        # After applying attention:
        # - attn_windows retains the shape (num_windows, window_area, channels) where `window_area` is the product of window dimensions (e.g., window_d * window_h * window_w for 3D).
        # - The attention mechanism is performed independently within each window, and the shape of the output remains consistent.

        # attn_windows has shape (num_windows, window_area, channels) after attention
        # Reverse window partitioning by reshaping attention windows back to original shape per window
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        # After reshaping:
        # - attn_windows now has shape (num_windows, window_d, window_h, window_w, channels) for 3D input

        # Restore spatial structure of the original input using window reversing
        shifted_x = window_reverse(attn_windows, window_size, dims)
        # After window_reverse:
        # - shifted_x is restored to its spatial structure with shape (batch_size, depth, height, width, channels) for 3D


        # If a shift was applied earlier, roll back the shift to restore original structure
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                # Apply positive rolling shift to revert the previous shift operation (REVERSE CYCLE SHIFT)
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            # No shift was applied, retain the tensor as is
            x = shifted_x

        # Remove padding if any was applied earlier to restore original dimensions
        if len(x_shape) == 5:
            # Check if any padding was added in depth, height, or width dimensions
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                # Slice out the padded regions to return to original dimensions
                x = x[:, :d, :h, :w, :].contiguous()

        return x  # Return the processed tensor


    def forward_part2(self, x):
        """
        Applies a series of operations on the input tensor 'x', including:
          - 1. Normalization
          - 2. Multi-layer perceptron (MLP) Transformation
          - 3. Drop Path Regularization
        This function contributes to the forward pass (second part) of the Swin Transformer block.

        Args:
            x (Tensor): Input tensor to be processed.

        Returns:
            Tensor: Output tensor after applying normalization, MLP, and drop path regularization.
        """
        # Normalize the input tensor 'x' across the last dimension
        # Layer normalization helps stabilize and optimize the learning process
        x = self.norm2(x)

        # Pass the normalized tensor through a Multi-Layer Perceptron (MLP)
        # The MLP typically includes linear transformations, non-linear activations, and optional dropout
        x = self.mlp(x)

        # Apply drop path regularization (also known as stochastic depth)
        # Drop path regularization randomly drops entire layers or paths during training to improve generalization
        # In this case, if dropped, the output of this function will be nullified, effectively skipping the contribution of this part during training
        return self.drop_path(x)



    def load_from(self, weights, n_block, layer):
        """
        Load weights from a pre-trained Swin Transformer model.
        This method copies specific parameters from a state_dict into the corresponding layers of this block.
        """
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            # Copy each relevant parameter from the weights
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj

    def forward(self, x, mask_matrix):
        """
        Forward pass of the SwinTransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            mask_matrix (torch.Tensor): Attention mask matrix to restrict computations.

        Returns:
            torch.Tensor: Output tensor after applying the Swin Transformer block.
        """
        # Save a copy of the input tensor as a shortcut (residual connection)
        shortcut = x

        # Check if gradient checkpointing is used; if so, compute the first part of the forward pass
        if self.use_checkpoint:
            # Use PyTorch's checkpointing to save memory during training by re-computing forward pass during backward pass
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            # Regular first part of the forward pass without checkpointing
            x = self.forward_part1(x, mask_matrix)

        # Add the residual connection and apply drop path regularization
        x = shortcut + self.drop_path(x)

        # Perform the second part of the forward pass, with optional checkpointing for memory savings
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            # If not using checkpointing, directly compute the second part of the forward pass
            x = x + self.forward_part2(x)

        return x


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    This class implements a single stage of Swin Transformer, which operates on input data
    using window-based self-attention and shift mechanisms to enhance spatial interactions.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Initializes the BasicLayer for a Swin Transformer block stage.

        Args:
            dim (int): Number of feature channels for the input.
            depth (int): Number of Swin Transformer blocks in this stage.
            num_heads (int): Number of attention heads used in the multi-head attention module.
            window_size (Sequence[int]): Size of the local attention window (e.g., [7, 7, 7]).
            drop_path (list): List containing the stochastic depth rates for each block.
            mlp_ratio (float): Ratio of MLP hidden dimensions to the input dimension.
            qkv_bias (bool): If True, adds a learnable bias to query, key, and value tensors.
            drop (float): Dropout rate applied to MLP layers.
            attn_drop (float): Dropout rate applied to attention weights.
            norm_layer (LayerNorm): Normalization layer used in the blocks.
            downsample (nn.Module | None): Optional downsampling module applied at the end of the layer.
            use_checkpoint (bool): If True, enables gradient checkpointing to reduce memory usage.
        """
        super().__init__()

        # Set window size, shift size, and no-shift size for the layer
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)  # Shift size is half of the window size
        self.no_shift = tuple(0 for i in window_size)  # No shift is represented by zeros
        self.depth = depth  # Number of Swin Transformer blocks in this stage
        self.use_checkpoint = use_checkpoint  # Use checkpointing for memory efficiency

        # Create a list of Swin Transformer blocks for this stage
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,  # Number of feature channels
                    num_heads=num_heads,  # Number of attention heads
                    window_size=self.window_size,  # Window size for attention
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,  # Alternate between no shift and shift
                    mlp_ratio=mlp_ratio,  # MLP hidden dimension ratio
                    qkv_bias=qkv_bias,  # Add bias to query, key, value tensors if True
                    drop=drop,  # Dropout rate for MLP
                    attn_drop=attn_drop,  # Dropout rate for attention
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,  # Drop path rate for stochastic depth
                    norm_layer=norm_layer,  # Normalization layer
                    use_checkpoint=use_checkpoint,  # Use gradient checkpointing
                )
                for i in range(depth)  # Create `depth` number of blocks
            ]
        )

        # Initialize optional downsampling module if provided
        self.downsample = downsample
        if callable(self.downsample):  # Check if downsample is callable
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x):
        """
        Forward pass through the BasicLayer.

        Args:
            x (Tensor): Input tensor of shape (b, c, d, h, w) for 3D data.

        Returns:
            Tensor: Output tensor after applying Swin Transformer blocks and optional downsampling.
        """
        # Get the shape of the input tensor (batch size, channels, depth, height, width)
        x_shape = x.size()

        # Handle 3D input tensors (e.g., volumetric data with batch size, channels, depth, height, width)
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape  # Extract dimensions from input shape

            # Determines the effective window_size and shift_size based on the dimensions
            # (d, h, w) of the input tensor and pre-defined window size and shift size values from the instance.

            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)

            x = rearrange(x, "b c d h w -> b d h w c")

            # dp, hp, and wp values are used to define the "padded" dimensions of the input,
            # ensuring compatibility with window-based operation
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]

            # The `compute_mask` function generates a mask that partitions the input tensor into distinct regions,
            # based on specified dimensions (`[dp, hp, wp]`), a `window_size`, and a `shift_size`.
            # This mask is used to control the attention mechanism during computation, ensuring that elements
            # only attend to others within the same window or shifted region.
            #
            # Key details:
            # - `dims = [dp, hp, wp]` are the padded input dimensions (depth, height, and width).
            # - `window_size` specifies the size of each window for partitioning, e.g., (7, 7, 7).
            # - `shift_size` specifies the window shift amount, used to enhance spatial interactions.
            # - `x.device` indicates where the computation occurs (CPU/GPU).
            #
            # The function divides the input tensor into regions, assigns a unique integer label to each region,
            # and creates an attention mask (`attn_mask`) by comparing labels within and across windows.
            # This mask restricts attention computations to only valid elements, optimizing efficiency
            # and adhering to the Swin Transformer's local attention mechanism.

            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)

            # Apply each Swin Transformer block in sequence to the input tensor
            for blk in self.blocks:
                x = blk(x, attn_mask)

            # Reshape the output back to its original shape with updated channels
            # Before reshaping, 'x' has dimensions (b, dp, hp, wp, c), where dp, hp, and wp may include padding
            # After reshaping, 'x' returns to its original spatial dimensions (b, d, h, w, -1), removing any padding
            # -1 means that the last dimension size is infered by dividing the total number of elements by (b * d * h * w).
            # It adjusts the feature dimension (channel dimension) to accommodate transformations done by the Swin Transformer blocks.
            x = x.view(b, d, h, w, -1)

            # Apply optional downsampling if a downsample module is defined
            if self.downsample is not None:
                x = self.downsample(x)

            # Rearrange tensor back to original format (b, c, d, h, w)
            x = rearrange(x, "b d h w c -> b c d h w")

        else:
            # Raise an error if the input tensor does not have the expected number of dimensions
            raise ValueError("Unsupported dimensions. Expected input to have length of 5 dimensions (b, d, h, w, c).")

        return x  # Return the processed tensor


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        else:
          raise ValueError(f"expecting 3D dim, got {dim}.")
    def forward(self, x):

        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 0::2, 1::2, :]
            x5 = x[:, 1::2, 1::2, 0::2, :]
            x6 = x[:, 0::2, 1::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)

        else:
          raise ValueError(f"expecting 5D x, got {x.shape}.")

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
    ) -> None:
        """
        Initializes the Swin Transformer model.

        Args:
            in_chans: Number of input channels.
            embed_dim: Dimension of linear projection output channels.
            window_size: Local window size used for window-based attention.
            patch_size: Size of input patches.
            depths: Number of layers in each transformer stage.
            num_heads: Number of attention heads in each stage.
            mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
            qkv_bias: Boolean indicating whether to add a learnable bias to query, key, and value tensors.
            drop_rate: Dropout rate applied to the input embeddings.
            attn_drop_rate: Dropout rate specific to the attention mechanism.
            drop_path_rate: Rate for stochastic depth (drop path).
            norm_layer: Normalization layer type.
            patch_norm: Whether to add normalization after patch embedding.
            use_checkpoint: Enables gradient checkpointing for reduced memory usage.
            spatial_dims: Number of spatial dimensions (e.g., 3 for 3D data).
            downsample: Module used for downsampling between stages.
            use_v2: Boolean indicating whether to use an updated version with residual convolutional blocks.
        """
        super().__init__()
        self.num_layers = len(depths)  # Number of stages in the transformer
        self.embed_dim = embed_dim  # Embedding dimension size
        self.patch_norm = patch_norm  # Whether to normalize after patch embedding
        self.window_size = window_size  # Size of the attention window
        self.patch_size = patch_size  # Patch size for embedding input

        # Initialize the patch embedding layer
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # Apply normalization if specified
            spatial_dims=spatial_dims,
        )

        # Dropout applied to positionally encoded input embeddings
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Calculate drop path rate schedule for each layer using linear interpolation
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.use_v2 = use_v2  # Boolean to check if using version 2
        self.layers1 = nn.ModuleList()  # List for the first stage's layers
        self.layers2 = nn.ModuleList()  # List for the second stage's layers
        self.layers3 = nn.ModuleList()  # List for the third stage's layers
        self.layers4 = nn.ModuleList()  # List for the fourth stage's layers

        # If using version 2, initialize additional layers with residual convolutional blocks
        if self.use_v2:
            self.layers1c = nn.ModuleList()
            self.layers2c = nn.ModuleList()
            self.layers3c = nn.ModuleList()
            self.layers4c = nn.ModuleList()

        # Set up the downsampling module
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample

        # Loop through each stage and initialize layers
        for i_layer in range(self.num_layers):
            # Create a BasicLayer instance for each stage
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),  # Double the dimension for each stage
                depth=depths[i_layer],  # Number of layers in this stage
                num_heads=num_heads[i_layer],  # Number of attention heads
                window_size=self.window_size,  # Size of the attention window
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],  # Drop path rate for this stage
                mlp_ratio=mlp_ratio,  # MLP hidden dimension to embedding dimension ratio
                qkv_bias=qkv_bias,  # Bias for query, key, and value tensors
                drop=drop_rate,  # General dropout rate
                attn_drop=attn_drop_rate,  # Attention dropout rate
                norm_layer=norm_layer,  # Normalization layer type
                downsample=down_sample_mod,  # Downsampling module
                use_checkpoint=use_checkpoint,  # Enable gradient checkpointing for memory efficiency
            )

            # Append the layer to the appropriate module list
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)

            # Add corresponding residual convolutional layers if using version 2
            if self.use_v2:
                layerc = UnetrBasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=embed_dim * 2**i_layer,
                    out_channels=embed_dim * 2**i_layer,
                    kernel_size=3,
                    stride=1,
                    norm_name="instance",
                    res_block=True,
                )
                if i_layer == 0:
                    self.layers1c.append(layerc)
                elif i_layer == 1:
                    self.layers2c.append(layerc)
                elif i_layer == 2:
                    self.layers3c.append(layerc)
                elif i_layer == 3:
                    self.layers4c.append(layerc)

        # Calculate the number of features after the final stage
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        """
        Applies projection and optional normalization to the input tensor.

        Args:
            x (Tensor): Input tensor.
            normalize (bool): Whether to apply layer normalization.

        Returns:
            Tensor: Projected and normalized tensor.
        """
        if normalize:
            x_shape = x.shape
            ch = int(x_shape[1])  # Number of channels
            if len(x_shape) == 5:  # If input is 5D (3D spatial data)
                x = rearrange(x, "b c d h w -> b d h w c")  # Rearrange dimensions for normalization
                x = F.layer_norm(x, [ch])  # Apply layer normalization
                x = rearrange(x, "b d h w c -> b c d h w")  # Rearrange back to original dimensions
            else:
                # Handle other input dimensions (e.g., 2D data)
                raise ValueError("Unsupported dimensions. Expected input to have length of 5 dimensions (b, d, h, w, c).")
        return x

    def forward(self, x, normalize=True):
        """
        Forward pass for the Swin Transformer.

        Args:
            x (Tensor): Input tensor.
            normalize (bool): Whether to apply normalization after projection.

        Returns:
            List[Tensor]: Output tensors from each stage.
        """
        # Apply patch embedding and dropout to the input
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)

        # Apply projection and normalization
        x0_out = self.proj_out(x0, normalize)

        # Forward pass through each stage, conditionally using residual convolutional layers if specified
        if self.use_v2:
            x0 = self.layers1c[0](x0.contiguous())
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        if self.use_v2:
            x1 = self.layers2c[0](x1.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        if self.use_v2:
            x2 = self.layers3c[0](x2.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        if self.use_v2:
            x3 = self.layers4c[0](x3.contiguous())
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)

        # Return outputs from each stage
        return [x0_out, x1_out, x2_out, x3_out, x4_out]


class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        use_v2=False,
        **kwargs
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dimensions (e.g. 3 for 3D data).
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"`.
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beginning of each swin stage.
        """

        super().__init__()

        self.architecture_config = kwargs["architecture_config"]
        self.attention_config = kwargs["attention_config"]
        self.normalization_config = kwargs["normalization_config"]
        self.regularization_config = kwargs["regularization_config"]

        # Ensure sizes match spatial dimensions
        # patch_size = ensure_tuple_rep(2, spatial_dims)
        # window_size = ensure_tuple_rep(7, spatial_dims)
        patch_size = self.architecture_config["patch_embedding_size"]
        window_size = self.attention_config["window_size"]

        # Ensure valid spatial dimensions
        if not (spatial_dims == 3):
            raise ValueError("Spatial dimension should be 3.")

        # Validate rates between 0 and 1
        if not (0 <= drop_rate <= 1):
            raise ValueError("Dropout rate should be between 0 and 1.")
        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("Attention dropout rate should be between 0 and 1.")
        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("Drop path rate should be between 0 and 1.")

        # Ensure feature size is divisible by 12 for the multi-head attention mechanism
        if feature_size % 12 != 0:
            raise ValueError("Feature size should be divisible by 12.")

        self.normalize = normalize

        # Define the Swin Transformer-based encoder (`SwinTransformer`) to be used in this UNETR
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=self.attention_config["window_size"],
            patch_size=self.architecture_config["patch_embedding_size"],
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=self.architecture_config["mlp_ratio"],
            qkv_bias=self.attention_config["qkv_bias"],
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            patch_norm=self.normalization_config["patch_norm_in_swinViT"],
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
            use_v2=self.architecture_config["use_SWIN_v2"],
        )

        # Encoder stages - Use `UnetrBasicBlock` for encoding input features
        # These layers transform the input tensor into feature maps at different resolutions
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # Decoder stages - Use `UnetrUpBlock` for decoding feature maps to higher resolutions
        # These layers upsample the feature maps to reconstruct the original spatial resolution
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        # Output block that takes the final feature map and converts it to desired output channels
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def load_from(self, weights):
        # Load weights from a pretrained model
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            # Load weights for downsampling layers and other components
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            # Repeating for layers3 and layers4

    def forward(self, x_in):
        """
        Forward pass for the SwinUNETR model.

        Args:
            x_in: Input tensor, typically with shape (batch, channels, depth, height, width).

        Returns:
            logits: Output predictions after applying the Swin Transformer and decoding layers.
        """

        # Pass input through the Swin Transformer encoder
        hidden_states_out = self.swinViT(x_in, self.normalize)

        # Apply encoder blocks to extract features at multiple resolutions
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])

        # Apply the decoder blocks in a hierarchical manner to reconstruct the image
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)

        # Generate final output from the last upsampled decoder output
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)

        return logits


class SwinUNETR_AIRT(torch.nn.Module):
    def __init__(self, architecture_config, attention_config, normalization_config, regularization_config):
        super(SwinUNETR_AIRT, self).__init__()

        self.architecture_config = architecture_config
        self.attention_config = attention_config
        self.normalization_config = normalization_config
        self.regularization_config = regularization_config

        self.ensure_all_dimensions_divisible(self.architecture_config["model_input_dimensions"])
        
        self.model = SwinUNETR(
            use_v2=self.architecture_config["use_SWIN_v2"],
            in_channels = self.architecture_config["model_input_channels"],
            out_channels = self.architecture_config["model_output_channels"],
            depths = self.architecture_config["num_swin_transformer_blocks_in_layers"],
            num_heads = self.attention_config["heads"],
            feature_size = self.architecture_config["initial_feature_embedding_size"],
            norm_name = self.normalization_config["unet_block_norm_type"],
            drop_rate = self.regularization_config["transformer_block_drop_rate"],
            attn_drop_rate = self.attention_config["drop_rate"],
            dropout_path_rate = self.regularization_config["transformer_block_residual_block_dropout_path_rate"],
            normalize = self.normalization_config["use_norm_in_swinViT_after_layer"],
             # Architecture Configuration
            architecture_config=self.architecture_config,            
            # Attention Configuration
            attention_config=self.attention_config,
            # Normalization Configuration
            normalization_config=self.normalization_config,         
            # Regularization Configuration
            regularization_config=self.regularization_config
            
        )

        self.conv_layer = nn.Conv3d(self.architecture_config["model_input_dimensions"][2], 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # batch_size, in_channels, height, width, depth
        x = x.permute(0, 1, 4, 2, 3)
        # batch_size, in_channels, depth, height, width
        x = self.model(x)
        # batch_size, out_classes, depth, height, width
        x = x.permute(0, 2, 1, 3, 4)
        # batch_size, depth, out_classes, height, width
        x = self.conv_layer(x)
        # batch_size, 1, out_classes, height, width
        x = x.squeeze(1)
        # batch_size, out_classes, height, width
        return x

    def ensure_all_dimensions_divisible(self, input_dimensions, divisor=32):
        """
        Ensure that all dimensions in input_dimensions are divisible by the given divisor.
        If not, raise an exception.

        Args:
            input_dimensions (tuple): Input dimensions (e.g., (128, 128, 128)).
            divisor (int): The number to ensure divisibility by (default is 32).

        Raises:
            ValueError: If any dimension is not divisible by the divisor.
        """
        for dim in input_dimensions:
            if dim % divisor != 0:
                raise ValueError(
                    f"Dimension {dim} is not divisible by {divisor}. All dimensions must be divisible."
                )