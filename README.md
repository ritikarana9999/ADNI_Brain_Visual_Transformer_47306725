ADNI_Brain_Visual_Transformer_47306725
# Visual Transformer for the Classification of Alzheimer's Disease

Ritika Rana (Student Noumber : 47306725)

## Overview
The rapidly expanding field of medical imaging leverages deep learning and neural networks to enhance diagnostic accuracy and reduce human errors in healthcare and medicine. This project seeks to create a Vision Transformer. The objective is to enable the Vision Transformer to categorize MRI brain scans as either indicative of Alzheimer's Disease or Normal Cognitive function as Vision Transformers can achieve excellent results compared to convolutional networks while demanding fewer training resources. The target for this model is a testing accuracy of at least 80%.

## Vision Transformers
Conventional Transformers employ the self-attention mechanism to discern the interplay among multiple inputs. Referred to as ViT (Vision Transformers), these models take the original Transformer architecture and adapt the attention mechanism to process 2D images, making them suitable for classification tasks.

<p align="center">
    <img src="resources/vit.gif" alt="Vision Transformer Architecture">
</p>

You can input 2D images into the Vision Transformer (ViT), and the images are divided into smaller patches. These patches are then transformed into 1D vectors through linear projections. Learnable class embeddings can be included, and to maintain the patch ordering, positional encodings can be introduced. These flattened patches, enriched with class or positional embeddings, are subsequently processed by conventional Transformer encoders to identify relationships within the image data patches. Finally, a Multi-Layer Perceptron (MLP), a neural network capable of learning relationships, is employed for the classification of inputs.
Components in a Transformer Encoder are as follows:


##Project Requirements 
The following dependencies were likely to provide the most reliable performance:

- Python version 3.10.4
- Tensorflow version 2.10.0, which is employed for tasks such as data loading, model construction, training, and prediction.

- Tensorflow Addons version 0.18.0, specifically utilized to implement the 'AdamW' optimizer for the model.

- Matplotlib version 3.5.2, used for generating visualizations, including plots for model loss and accuracy, as well as confusion matrices for evaluating test results.

##Overview of the Repository

The document has the following sections:

The resources/ directory houses the images utilized in this README document.

"Parameters" includes the hyperparameters employed to define data loading and model configurations.

"Modules" has the elements that make up the Vision Transformer.

"Dataset" contains the function responsible for data loading.

"Train" can access the functions for assembling and training the model.

"Predict" comprises the functions required for making predictions using the trained model.

Lastly, "Utils" encompasses functions dedicated to data visualization.

Setting Parameters
Prior to initiating model training, make sure to configure the global variables located in parameters.py. The variables that require configuration include:

IMG_SIZE: The height and width of the image.
PATCH_SIZE: The height and width of individual patches.
BATCH_SIZE: The batch size for both training and testing data.
PROJECTION_DIM: The dimensions of the attention layer.
LEARN_RATE: The learning rate for the optimizer.
ATTENTION_HEADS: The number of attention heads.
DROPOUT_RATE: The fraction of units to drop out in each Dense Layer (expressed as a decimal).
TRANSFORMER_LAYERS: The quantity of transformer encoders to incorporate.
WEIGHT_DECAY: The weight decay used by the optimizer.
EPOCHS: The number of epochs for model training.
MLP_HEAD_UNITS: The number of units in the MLP head classifier for feature learning (specified as an array, with each element representing the units in an additional Dense Layer).
DATA_PATH: The path from which the dataset will be loaded.
MODEL_SAVE_DEST: The location to save the model.
The variables INPUT_SHAPE, HIDDEN_UNITS, and NUM_PATCHES are calculated automatically.

Building and Training the Model
After configuring the parameters, execute till train.  Upon completion of training, the model will be saved in HDF5 format at the directory specified by MODEL_SAVE_DEST. Additionally, loss and accuracy curves will be generated and saved in the working directory.

To make predictions using the trained model, run predict. This script loads the model from MODEL_SAVE_DEST and evaluates the test set. A confusion matrix will be automatically generated and saved in the working directory.

Dataset
we can download the original dataset from the ADNI website. To specify the data source path, modify the DATA_LOAD_DEST constant in parameters. The data is expected to adhere to the following file structure:

train/
    AD/
        images
        ...
    NC/
        images
        ...
test/
    AD/
        images
        ...
    NC/
        images
        ...

Given that this dataset exclusively contains two categories, it is loaded using binary encoding. In this scheme, the value 0 represents the AD class for brains afflicted with Alzheimer's Disease, and 1 is assigned to NC, signifying brains with normal cognitive function.

##Training, Validation and Test Splits
The provided dataset consisted of 21,520 images in the training folder, and 9000 images in the testing folder. Since Vision Transformers are very data-hungry models which require a lot of data for training, it was decided that the validation set would be split from half of the image data in the testing folder. This gives a dataset split of:

21,520 images in training set
4500 images in validation set
4500 images in testing set

Data Augmentation
The data loaded was also augmented before passing to the model for training and prediction.
Normalization, RandomFlip, RandomZoom and RandomRotate layers were applied to each image in the three datasets.
By augmenting the data, the model would be less likely to overfit on the already small set of training data. The randomized actions have all been passed a seed parameter to allow for the reproducibility of results.


##Vision Transformer Implementation
##Changes made to Original Vision Transformer
The architecture of the implemented Vision Transformer model exhibits slight deviations from the original model presented in the Vision Transformer paper (referenced as [1]).

The key distinctions include the incorporation of Shifted Patch Tokenization and the utilization of Local Self-Attention. This study demonstrates that integrating both Shifted Patch Tokenization and Local Self-Attention into a Vision Transformer addresses the absence of locality bias in Transformers, enhancing their ability to learn effectively, especially when working with smaller datasets. Given that our dataset comprises a relatively modest total of 30,520 images, these techniques were deemed essential additions to the model.

Shifted Patch Tokenization involves a slight shift of input images, either to the right or left, as well as up or down. This adjustment widens the receptive field of the Transformer, thereby enhancing its capability to discern the relationships between patches in the image.

Shifted Patch Tokenization

Local Self-Attention is similar to the traditional Multi-Head Attention (MHA) layer, however an additional diagonal mask is applied to shift the ViT's attention towards inter-token relationships rather than its own tokens. A Learnable Temperature Scaling is also included for the model to learn the temperature of the Softmax layer in the MHA layer automatically, helping to either sharpen score distribution or attention distribution.

Local Self-Attention

Additionally, the class token included in the Transformer Encoder from the original paper was also removed. An author from the original paper stated here that the class token is not important and unnecessary and thus, in our model, classification is done through the use of an MLP (Dense layers) which learn features.

Vision Transformer Architecture
After the initial input layer of the Vision Transformer, data is passed into a the PatchLayer layer, which splits the images into patches with height and width of the PATCH_SIZE constant. In this layer, Shifted Patch Tokenization is also applied to the inputted images before patching. The original image split into patches becomes:

MRI Patches

The images which have Shifted Patch Tokenization applied and are patched become:

MRI Patches LEFT UP MRI Patches LEFT DOWN

MRI Patches RIGHT UP MRI Patches RIGHT DOWN

From here, these images are flattened into vectors before being passed to the EmbedPatch layer, which is used for embedding patch positions to the flattened vectors. After patches have been encoded with their positions, the vectors are passed into the Transformer blocks. Within the Transformer blocks, an architecture similar to the original is followed, except Locality Self-Attention is applied together with the Multi-Head Attention layer, rather than just the Multi-Head Attention layer itself. Following the Transformer encoder blocks, MLP layers are used to learn features and to make the final classification (since the class token is no longer being used).

In testing, it was found that if the model returned logits and BinaryCrossEntropy was evaluated from logits, results were more stable. Thus, the final MLP classification layer does not have any activation applied.

Training and Testing Process
Multiple combinations of hyperparameters were tested and tuned, based on the results of training, validation and testing accuracies returned. Three models of varying complexity and their results are documented below.
