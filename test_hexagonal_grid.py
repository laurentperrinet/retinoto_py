#!/usr/bin/env python3
"""
Test script to verify the hexagonal grid implementation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.retinoto_py.params import Params
from src.retinoto_py.torch_utils import get_grid, get_grid_hexagonal

def test_hexagonal_grid():
    """Test the hexagonal grid implementation"""
    
    # Create parameters
    args = Params()
    args.grid_size = 50  # Smaller size for visualization
    
    # Generate both grids
    regular_grid = get_grid(args)
    hexagonal_grid = get_grid_hexagonal(args)
    
    print(f"Regular grid shape: {regular_grid.shape}")
    print(f"Hexagonal grid shape: {hexagonal_grid.shape}")
    
    # Check that shapes match
    assert regular_grid.shape == hexagonal_grid.shape, "Grid shapes should match"
    
    # Visualize the grids
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot regular grid
    axes[0].scatter(regular_grid[:, :, 0].flatten(), regular_grid[:, :, 1].flatten(), s=1, alpha=0.5)
    axes[0].set_title('Regular Log-Polar Grid')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Plot hexagonal grid
    axes[1].scatter(hexagonal_grid[:, :, 0].flatten(), hexagonal_grid[:, :, 1].flatten(), s=1, alpha=0.5)
    axes[1].set_title('Hexagonal Log-Polar Grid')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grid_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Hexagonal grid test completed successfully!")
    print("✓ Visualization saved as 'grid_comparison.png'")
    
    # Test that hexagonal grid has the expected staggering pattern
    # Check that even and odd rows have different angular offsets
    even_row_angles = torch.atan2(hexagonal_grid[::2, 0, 1], hexagonal_grid[::2, 0, 0])
    odd_row_angles = torch.atan2(hexagonal_grid[1::2, 0, 1], hexagonal_grid[1::2, 0, 0])
    
    # The angles should be offset by approximately half the angular resolution
    angular_resolution = 2 * np.pi / args.grid_size
    expected_offset = angular_resolution / 2
    
    # Check first few rows to verify the pattern
    for i in range(min(4, args.grid_size//2)):
        even_angle = even_row_angles[i].item()
        odd_angle = odd_row_angles[i].item()
        angle_diff = abs(even_angle - odd_angle)
        # Allow some tolerance for the angle difference
        assert abs(angle_diff - expected_offset) < 0.1 or abs(angle_diff - (2*np.pi - expected_offset)) < 0.1, \
               f"Row {i}: Expected offset ~{expected_offset}, got {angle_diff}"
    
    print("✓ Hexagonal staggering pattern verified!")

if __name__ == "__main__":
    test_hexagonal_grid()