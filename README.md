# DETAILED REPORT OF THIS PROJECT
# Title: Generating Images Using a GAN
 
## Introduction:

In this project, we aimed to generate new images of specific category(animals) using a Generative Adversarial Network (GAN). The CIFAR-10 dataset was chosen as it offers a diverse set of animals images, and we have a sufficient number of samples (60,000) to train our model without the need for extensive data augmentation.

## Dataset:

 used the CIFAR-10 dataset, which consists of 60,000 32x32 color images across  different classes of animals, with each class containing 6,000 images. This dataset was selected due to its suitability for training a GAN and its relatively low image resolution, reducing computational requirements.(considering training images with low number of images leads to inefficiency)

## Data Preprocessing:

Before training the GAN, performed essential data preprocessing steps:
Normalization:  normalized pixel values to the range [-1, 1].
Resizing: The real images and generated images were resized to 299*299*3 to match with the inception model(model used for calculating FID and IS score)

## Model Architecture:

### **1)Generator:**

#### Generator Architecture:

The generator is a critical component of the Generative Adversarial Network (GAN) responsible for generating synthetic images. It transforms a latent vector (random noise) into an image that should resemble real images from the dataset.The generator architecture consists of dense and transposed convolutional layers that gradually upsample the initial 8x8 structure to a 32x32 image. The choice of hyperparameters, activation functions, and layer configurations is based on best practices in GAN architecture design. The generator is not compiled separately since it is trained as part of the GAN combined model.

#### Layer 1: Dense Layer

**Input:** The generator starts with a latent vector of dimension latent_dim (e.g., 100).

**Neurons:** The dense layer has n_nodes neurons, calculated as 128 * 8 * 8, which equals 8192 nodes.

**Activation Function:** Leaky ReLU (Rectified Linear Unit) with a small alpha value of 0.2 is used as the activation function. Leaky ReLU helps prevent the vanishing gradient problem and allows some small negative values to pass, which can be important for learning.

#### Layer 2: Reshape Layer

The output from the dense layer is reshaped into a 3D tensor with dimensions 8x8x128. This forms the initial 8x8 image-like structure from the latent vector.

#### Layer 3: Transposed Convolutional Layer

**Filters:**A transposed convolutional layer with 128 filters and a kernel size of 4x4 is used.

**Strides:** A stride of 2x2 is applied to upsample the image to 16x16.

**Padding:** 'Same' padding is used to maintain spatial dimensions.

**Activation Function:** Leaky ReLU with an alpha value of 0.2 is applied again.

#### Layer 4: Transposed Convolutional Layer

**Filters:** Another transposed convolutional layer with 128 filters and a kernel size of 4x4 is employed.

**Strides:** A stride of 2x2 is used for further upsampling to 32x32.

**Padding:** 'Same' padding is retained.

**Activation Function:** Leaky ReLU with an alpha value of 0.2 is applied.

#### Layer 5: Output Layer

**Neurons:** The final output layer comprises 3 neurons, corresponding to the RGB color channels (Red, Green, Blue).

**Activation Function:** Hyperbolic Tangent (tanh) activation is selected for the output layer, ensuring that pixel values are within the range [-1, 1]. This matches the normalization used in preprocessing real images.



### 2)**Discriminator:**

#### Discriminator Architecture:

The discriminator is a key component of the Generative Adversarial Network (GAN) responsible for distinguishing between real and generated (fake) images. 
The discriminator architecture comprises convolutional layers for feature extraction, dropout for regularization, and a sigmoid output layer for binary classification. The choice of hyperparameters and activation functions is based on best practices in GAN architecture design. This discriminator will be trained in an adversarial setting alongside the generator to improve its ability to distinguish real from fake images.The architecture is designed with multiple layers to extract features from images effectively.

#### Layer 1: Convolutional Layer

**Filters:** 128 filters of size 3x3 are employed in the first convolutional layer.

**Strides:** A stride of 2x2 is used to downsample the image spatially, reducing its dimensions.

**Padding:** 'Same' padding is applied to ensure the output size matches the input size.

**Activation Function:** Leaky ReLU (Rectified Linear Unit) with a small alpha value of 0.2 is used as the activation function. Leaky ReLU helps prevent the vanishing gradient problem and allows some small negative values to pass, which can be important for learning.

#### Layer 2: Convolutional Layer

**Filters:** Another 128 filters of size 3x3 are utilized in the second convolutional layer.

**Strides:** Similar to the first layer, a stride of 2x2 is applied for downsampling.

**Padding:** 'Same' padding is maintained for consistency.

**Activation Function:** Leaky ReLU with an alpha value of 0.2 is used again.

#### Layer 3: Flattening

The output from the second convolutional layer is flattened into a 1D vector.
This step prepares the data for feeding into a dense (fully connected) layer.

#### Layer 4: Dropout

A dropout layer with a rate of 0.4 is included.
Dropout is employed to mitigate overfitting by randomly deactivating 40% of the neurons during training.

#### Layer 5: Output Layer

**Neurons:** A single neuron is used in the output layer, as the task is binary classification (real or fake).

**Activation Function:** Sigmoid activation is employed, ensuring the output is within the range [0, 1]. This represents the discriminator's confidence in the input image being real (closer to 1) or fake (closer to 0).

#### Model Compilation:

**Optimizer:** The Adam optimizer is selected with a learning rate (lr) of 0.0002 and a momentum term (beta_1) of 0.5.

**Loss Function:** Binary cross-entropy loss is chosen as the loss function, suitable for binary classification problems.

**Metrics:** The model is evaluated based on accuracy during training, although it's not the primary metric for GANs.


## GAN Architecture:

The GAN architecture is a composite model that brings together the generator and discriminator. This architecture facilitates the training process, where the generator aims to produce realistic images that can fool the discriminator, while the discriminator aims to distinguish between real and generated images.

#### Discriminator Trainability:

The discriminator is set to be not trainable within the GAN. This configuration is crucial to ensure that during GAN training, only the generator's weights are updated, while the discriminator's weights remain fixed. This separation of training allows the generator to learn and improve without interference from the discriminator's feedback.

#### Layer 1: Generator

The GAN begins with the generator, which is responsible for producing synthetic images.
The output of the generator is a generated image with the same dimensions as the real images in the dataset (e.g., 32x32x3).

#### Layer 2: Discriminator

The generator's output is then fed into the discriminator, which evaluates the authenticity of the generated image.
The discriminator's task is to classify whether the input image is real or fake (generated).
During the training of the GAN, the discriminator's weights are not updated, ensuring that it remains a fixed benchmark for the generator to improve upon.

#### Layer 3: Compilation

The GAN is compiled with the following configurations:

**Loss Function:** Binary Cross-Entropy
Binary cross-entropy is commonly used in GANs as it measures the difference between the predicted (discriminator's output) and target labels (real or fake).

**Optimizer:** Adam
The Adam optimizer is chosen with a learning rate (lr) of 0.0002 and a beta_1 value of 0.5. These hyperparameters are standard choices for GAN training.


## Model Training

Training the Generator and Discriminator in a GAN:

The training process for a Generative Adversarial Network (GAN) is a crucial aspect of achieving high-quality generated images. The provided code outlines the training procedure for the generator (G), discriminator (D), and the combined GAN model.

**Epochs and Batches:**

The training process is divided into a specified number of epochs, which determine the number of times the entire dataset is used for training.
Within each epoch, the dataset is further divided into batches, with each batch containing a fixed number of samples.

**Discriminator Training:**

The training begins with the discriminator (D). The primary goal of the discriminator is to distinguish between real and generated images.
In each batch, a random selection of real images from the dataset is used. These real images serve as the benchmark for the discriminator.
The weights of the discriminator are updated using the train_on_batch method with the real images, and the associated labels indicating that these are real samples.
The loss incurred during this phase is captured as d_loss_real.

**Generator Training:**

Following the discriminator training, the focus shifts to the generator (G).
The generator is tasked with creating synthetic images that resemble real images.
In each batch, a set of fake images is generated by the generator. These generated images represent the GAN's current attempt at creating realistic content.
The weights of the discriminator are again updated using the train_on_batch method with the generated images and labels indicating that these are fake samples.
The loss incurred during this phase is captured as d_loss_fake.

**Combined Loss:**

The GAN training involves optimizing both the generator and discriminator simultaneously.
The generator aims to minimize the discriminator's ability to distinguish fake images from real ones.
The loss incurred by the generator is calculated during this phase and captured as g_loss.
The generator uses the discriminator's feedback (discriminator's loss) to enhance its ability to generate more convincing images.

**Adversarial Training:**

The GAN leverages adversarial training, where the generator and discriminator engage in a competitive learning process.
The generator continually strives to create images that can fool the discriminator, while the discriminator becomes more adept at distinguishing real from fake.
Monitoring Progress:

The training process prints and logs important metrics, including the discriminator's loss on real and fake samples, and the generator's loss.
These metrics are valuable for tracking the progress of the GAN during training.

Epoch Progress:

The code provides updates on the training progress after each batch within an epoch.
This iterative approach allows for real-time feedback and insight into how the generator and discriminator are performing.

**Saving the Generator Model:**

After completing all epochs and batches, the trained generator model is saved for future use.


## Model Evaluation:

Evaluated the quality of the generated images using quantitative metrics:

#### 1)Frechet Inception Distance (FID): 
It measures the similarity between the distribution of real and generated images in feature space. A lower FID indicates better image quality.
#### 2)Inception Score (IS): 
It assesses the quality and diversity of generated images. Higher IS values are desirable.
Additionally, I performed qualitative evaluation by visually inspecting a sample of generated images.

**Generated Images:**
loaded a pre-trained GAN model from the file 'cifar_generator_50epochs.h5'.
generated a set of high-resolution images (X_generated) by sampling random noise (latent points) and passing them through the GAN model.
scaled the generated images to the [0, 1] range and convert them to uint8 format.

**Resize Generated Images:**
The generated images are resized to 299x299 pixels using a function called resize_images.

**Feature Extraction:**
loaded a pre-trained InceptionV3 model for feature extraction. This model is used to capture features from the images.
The generated images (X_generated_resized) and real images  are preprocessed using the preprocess_input function to make them compatible with the InceptionV3 model.
Features are then extracted from both sets of images using the InceptionV3 model, resulting in features_generated and features_real.

#### FID Calculation:
The function calculate_fid calculates the Fréchet Inception Distance (FID) between the features of generated and real images. FID measures the similarity between the distributions of features in both sets.
It computes the mean (mu) and covariance (sigma) of the feature vectors for both sets.
It calculates the covariance square root and ensures it's real-valued.
Finally, it computes the FID score using the formula involving mean differences and the trace of the covariance matrices.

#### IS Calculation:
The function calculate_inception_score calculates the Inception Score (IS) for the generated images. IS measures the quality and diversity of generated images.
It first calculates the marginal probability distribution p(y) of the labels (classes) based on the generated images' features (p_yx).
Then, it computes the KL divergence for each image between p(yx) and p(y) and averages them.
Finally, it exponentiates the result to get the IS score.



## Deployment Architecture (Basic One)

**1)Choose a Cloud Provider:** Pick a cloud provider like AWS, Google Cloud, or Microsoft Azure.

**2)Create an API:** Use a framework like Flask (Python) to create a simple API.

**3)pload Model:** Upload your trained GAN model weights to a cloud storage service like Amazon S3, Google Cloud Storage, or Azure Blob Storage.

**4)API Endpoint:** Within your API code, load the GAN model from the cloud storage using its URL or API. This is a lightweight way to access your model without deploying it directly.

**5)Generate Images:** Set up an API route that takes parameters, such as the number of images to generate, and use the loaded GAN model to create and return these images.

**6)Deploy API:** Deploy your Flask API to a cloud service. Many cloud platforms offer serverless options that are simple to deploy and scale, such as AWS Lambda, Google CloudFunctions, or Azure Functions.

**7)Documentation:** Create clear documentation on how developers can use your API, including endpoints, parameters, and expected responses.

**8)Scale:** Cloud providers automatically handle scaling based on demand, so no need to worry about it.

**9)Cost Monitoring:** Keep an eye on the cost, as real-time image generation may consume resources.
