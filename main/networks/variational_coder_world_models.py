# Input shape: (64x64x3)
# Using 0 to 1 float instead of 0 to 255 int for RGB
# Coder parameter count: 4,446,915

# // Encode
# Conv(32, 4, stride=2), relu, # 31,31,32
# Conv(64, 4, stride=2), relu, # 14, 14, 64
# Conv(128, 4, stride=2), relu, # 6, 6, 128
# Conv(256, 4, stride=2), relu, # 2, 2, 256
 
# mu is one half of the output
# sigma is the other half
# latent vector is mu + sigma*randNormal(0,1)
# latent shape of 1,1, 1024

# "factored Gaussian distribution" is the randNormal, I need to make sure factored doesnt mean something special

# // Decode
# Deconv(128, 5, stride=2), relu, # 5,5,128
# Deconv(64, 5, stride=2), relu, # 13,13, 64
# Deconv(32, 6, stride=2), relu, # 30, 30, 32
# Deconv(3, 6, stride=2), sigmoid, # 64, 64, 3

# L2 distance as loss "in addition to KL divergence"?

# Our latent vector $z_t$ is sampled from a factored Gaussian distribution $N(\mu_t, \sigma_t^2 I)$, with mean $\mu_t\in \mathbb{R}^{N_z}$ and diagonal variance $\sigma_t^2 \in \mathbb{R}^{N_z}$. As the environment may give us observations as high dimensional pixel images, we first resize each image to 64x64 pixels and use this resized image as V's observation. Each pixel is stored as three floating point values between 0 and 1 to represent each of the RGB channels. The ConvVAE takes in this 64x64x3 input tensor and passes it through 4 convolutional layers to encode it into low dimension vectors $\mu_t$ and $\sigma_t$. 

# In the Car Racing task, $N_z$ is 32 
# while for the Doom task $N_z$ is 64. 

# The latent vector $z_t$ is passed through 4 of deconvolution layers used to decode and reconstruct the image.

# Each convolution and deconvolution layer uses a stride of 2. The layers are indicated in the diagram in Italics as Activation-type Output Channels x Filter Size. All convolutional and deconvolutional layers use relu activations except for the output layer as we need the output to be between 0 and 1. We trained the model for 1 epoch over the data collected from a random policy, using $L^2$ distance between the input image and the reconstruction to quantify the reconstruction loss we optimize for, in addition to KL loss.