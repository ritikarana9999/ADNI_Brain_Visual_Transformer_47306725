ADNI_Brain_Visual_Transformer_47306725
# Visual Transformer for the Classification of Alzheimer's Disease

Ritika Rana (Student Noumber : 47306725)

## Overview
The rapidly expanding field of medical imaging leverages deep learning and neural networks to enhance diagnostic accuracy and reduce human errors in healthcare and medicine. In 2020, a groundbreaking paper titled "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" demonstrated that traditional Convolutional Neural Networks were not indispensable. It revealed that Vision Transformers can achieve excellent results compared to state-of-the-art convolutional networks while demanding fewer training resources. This project seeks to create a Vision Transformer akin to the one detailed in the original paper. Its objective is to enable the Vision Transformer to categorize MRI brain scans as either indicative of Alzheimer's Disease or Normal Cognitive function. The target for this model is a testing accuracy of at least 80%.

## Vision Transformers
Conventional Transformers employ the self-attention mechanism to discern the interplay among multiple inputs. Referred to as ViT (Vision Transformers), these models take the original Transformer architecture and adapt the attention mechanism to process 2D images, making them suitable for classification tasks.

<p align="center">
    <img src="resources/vit.gif" alt="Vision Transformer Architecture">
</p>

You can input 2D images into the Vision Transformer (ViT), and the images are divided into smaller patches. These patches are then transformed into 1D vectors through linear projections. Learnable class embeddings can be included, and to maintain the patch ordering, positional encodings can be introduced. These flattened patches, enriched with class or positional embeddings, are subsequently processed by conventional Transformer encoders to identify relationships within the image data patches. Finally, a Multi-Layer Perceptron (MLP), a neural network capable of learning relationships, is employed for the classification of inputs.
Components in a Transformer Encoder are as follows:
