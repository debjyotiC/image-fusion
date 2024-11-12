import fusionapi
from fusionapi import cv2
import matplotlib.pyplot as plt

# Load images
visible_image = cv2.imread("images/optical_image.jpg")
thermal_image = cv2.imread("images/thermal_image.png")

# Create Gaussian and Laplacian pyramids for both images
levels = 2
gaussian_pyr_vis = fusionapi.gaussian_pyramid(visible_image, levels)
gaussian_pyr_therm = fusionapi.gaussian_pyramid(thermal_image, levels)
laplacian_pyr_vis = fusionapi.laplacian_pyramid(gaussian_pyr_vis)
laplacian_pyr_therm = fusionapi.laplacian_pyramid(gaussian_pyr_therm)

# Fuse the Laplacian pyramids
fused_pyramid = fusionapi.fuse_pyramids(laplacian_pyr_vis, laplacian_pyr_therm)

# Reconstruct the fused image from the fused Laplacian pyramid
fused_image = fusionapi.reconstruct_from_pyramid(fused_pyramid)

cv2.imwrite('fused.png', fused_image)

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(1, 3)

# For Sine Function
axis[0].imshow(cv2.cvtColor(visible_image, cv2.COLOR_BGR2RGB))
axis[0].set_title("Visible")

# For Cosine Function
axis[1].imshow(cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB))
axis[1].set_title("Thermal")

# For Tangent Function
axis[2].imshow(cv2.cvtColor(fused_image, cv2.COLOR_BGR2RGB))
axis[2].set_title("Blended")

plt.tight_layout()
plt.savefig("images/test.png", dpi=300)
plt.show()



