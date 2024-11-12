import cv2

def gaussian_pyramid(image, levels):
    """Generate a Gaussian pyramid."""
    pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def laplacian_pyramid(gaussian_pyramid):
    """Generate a Laplacian pyramid from a Gaussian pyramid."""
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], gaussian_expanded)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # The smallest Gaussian level
    return laplacian_pyramid

def fuse_pyramids(laplacian_pyramid1, laplacian_pyramid2):
    """Fuse two Laplacian pyramids by taking the average at each level."""
    fused_pyramid = []
    for lap1, lap2 in zip(laplacian_pyramid1, laplacian_pyramid2):
        fused = cv2.addWeighted(lap1, 0.5, lap2, 0.5, 0)
        fused_pyramid.append(fused)
    return fused_pyramid

def reconstruct_from_pyramid(laplacian_pyramid):
    """Reconstruct the image from its Laplacian pyramid."""
    image = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        image = cv2.pyrUp(image, dstsize=size)
        image = cv2.add(image, laplacian_pyramid[i])
    return image