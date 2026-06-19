import torch


def extract_spatial_histogram(image_tensor, grid_size=4, bins=10):
    """
    Extracts a block-wise histogram from a 3x224x224 image tensor.
    Returns a flattened 1D tensor of shape (grid_size * grid_size * bins).
    """
    # 1. Convert to Grayscale (Luminance)
    # Using the standard ITU-R 601 mathematical weights for RGB -> Grayscale
    r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
    gray_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b  # Shape: [224, 224]
    
    # 2. Calculate the size of each block
    _, height, width = image_tensor.shape
    h_step = height // grid_size  # 224 // 4 = 56
    w_step = width // grid_size   # 224 // 4 = 56
    
    block_histograms = []
    pixels_per_block = h_step * w_step  # 56 * 56 = 3136 pixels
    
    # 3. Loop through the grid
    for row in range(grid_size):
        for col in range(grid_size):
            
            # Calculate the exact bounding box for this specific block
            y_start, y_end = row * h_step, (row + 1) * h_step
            x_start, x_end = col * w_step, (col + 1) * w_step
            
            # Slice the block out of the grayscale image
            block = gray_tensor[y_start:y_end, x_start:x_end]
            
            # 4. Calculate the 10-bin histogram for this block
            # Assuming the image tensor is normalized between 0.0 and 1.0
            hist = torch.histc(block, bins=bins, min=0.0, max=1.0)
            
            # 5. CRITICAL: Normalize the histogram!
            # Instead of counts (e.g., 2000 pixels), convert to percentages (e.g., 0.63)
            # This prevents the MLP gradients from exploding.
            hist_normalized = hist / pixels_per_block
            
            block_histograms.append(hist_normalized)
            
    # 6. Flatten the 16 separate 10-bin histograms into a single 160-element vector
    final_vector = torch.cat(block_histograms) # Shape: [160]
    
    return final_vector