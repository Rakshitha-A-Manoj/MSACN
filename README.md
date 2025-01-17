# MSACN: Multi-Scale Adaptive Convolutional Network

## Overview

MSACN (Multi-Scale Adaptive Convolutional Network) is a deep learning-based model designed for accurate classification of crop diseases. This project aims to improve classification performance by leveraging multi-scale feature extraction and adaptive convolutional mechanisms.

## Features

- **Multi-Scale Adaptive Convolution**: Enhances feature extraction across different spatial scales.
- **Improved Data Augmentation**: Handles class imbalances and improves generalization.
- **Optimized Regularization Techniques**: Prevents overfitting while maintaining high classification accuracy.
- **Comparative Analysis**: Benchmarks against other deep learning models such as MLP ANN, DMCNN, MConvExt, and MSCPNet.

## Dataset

This project utilizes publicly available datasets for crop disease classification:

- **[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)** (Corn, Strawberry, Tomato)
- **[Wheat Leaf Dataset](https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset)** (Wheat)
- **[Rice Disease Dataset](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)** (Rice)

## Installation

```bash
git clone https://github.com/your-repo/MSACN.git
cd MSACN
```

## Model Architecture

The MSACN model consists of the following components:

- **Multi-Scale Convolutional Blocks**
- **Adaptive Feature Selection**
- **Batch Normalization and Dropout**
- **Fully Connected Layers with Softmax Output**

## Training

To train the model, use the following command:

```bash
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adjusted learning rate
    loss='categorical_crossentropy',  # Adjust loss function as needed
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=callbacks,
    class_weight=class_weights
)
```

### Training Configuration

- Optimizer: Adam with cosine annealing learning rate scheduler
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy, Precision, Recall, F1-score

## Evaluation

```bash
test_loss, test_acc = model.evaluate(test_ds)
```

The results are compared using classification reports for each class.

## Results

| Model   | Accuracy | Precision | Recall | F1-score |
|---------|----------|----------|--------|----------|
| MSACN   | 98.60%   | 98.00%   | 98.00% | 98.00%   |
| MLP ANN | 10.00%   | 52.00%   | 95.00% | 95.00%   |
| DMCNN   | 95.00%   | 95.00%   | 51.00% | 35.00%   |
| MConvNeXt| 51.00%   | 26.00%   | 12.00% | 03.00%   |
| MSCPNet | 11.00%   | 01.00%   | 10.00% | 02.00%   |

## Usage

To classify new images:

```bash
python predict.py --image path/to/image.jpg --model best_model.pth
```

## Future Work

- Improve model interpretability.
- Expand dataset coverage.
- Deploy as a web-based classification tool.
