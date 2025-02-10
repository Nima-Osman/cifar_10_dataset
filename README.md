<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
</head>
<body>
    <h1>Neural Network-Based Image Classification Using CNNs</h1>
    <h2>Overview</h2>
    <p>This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is designed to recognize objects across ten categories by leveraging deep learning techniques.</p>
    
<h2>Objectives</h2>
    <p>The goal of this project is to develop a CNN-based classifier capable of object recognition using CIFAR-10. Various architectures, hyperparameters, and training techniques are explored to optimize performance. Additionally, data augmentation and regularization methods are applied to reduce overfitting.</p>
    
<h2>Dataset</h2>
    <p>The CIFAR-10 dataset consists of 60,000 color images, each measuring 32x32 pixels, categorized into ten different classes. The dataset is split into 50,000 training images and 10,000 test images.</p>
<p align="center">
<img src="images\dash_1.png"  alt="Centered Image"  width="500">
</p>

<h2>Methodology</h2>
    <h3>Data Preprocessing</h3>
    <p>To improve convergence, pixel values are normalized to the range of [0,1]. Additionally, data augmentation techniques, such as random horizontal flips, rotations, and shifts, are applied to improve generalization.</p>

[For code](Neural network python code.ipynb)

<h3>Model Architecture</h3>
    <p>The CNN model consists of convolutional layers with ReLU activation functions, followed by max-pooling layers to reduce dimensionality. Dropout layers are incorporated to prevent overfitting, and the final output layer uses a softmax activation function for multi-class classification.</p>
    
<h3>Hyperparameter Tuning</h3>
    <p>Different hyperparameters are tested, including filter sizes, learning rates, and optimizers. Smaller filter sizes improve generalization but may limit pattern recognition. Learning rates ranging from 0.1 to 0.0001 are tested, with 0.001 providing the best balance. The Adam optimizer is chosen for its adaptive learning rate, though SGD with momentum is also considered for further improvements.</p>
    
<h3>Evaluation Metrics</h3>
    <p>The primary evaluation metric is validation loss, which helps identify overfitting. Categorical cross-entropy is used as the loss function for multi-class classification. Additional metrics, such as accuracy, precision, recall, and F1-score, are also considered to assess performance.</p>
    
<h2>Key Findings</h2>
    <p>Reducing filter sizes enhances generalization but can limit the model’s ability to recognize patterns. Dropout layers are effective in preventing overfitting but require fine-tuning. The Adam optimizer provides efficient training, although SGD with momentum might yield better convergence in certain cases. Monitoring validation loss is more reliable than accuracy alone for detecting overfitting.</p>
    
<p>The classification report provides a detailed breakdown of precision, recall, and F1-score for each class. The model achieved an overall accuracy of approximately 85%, with some classes, such as trucks and airplanes, achieving higher accuracy than others. The confusion matrix highlights misclassifications, with some overlap between similar classes like cats and dogs. 
    
The graphical analysis of training and validation loss indicates that the model successfully learned patterns without severe overfitting. Accuracy curves demonstrate steady improvement across epochs. The final test image of a truck was correctly classified, verifying the model's ability to generalize to unseen data.</p>

 <h2>Future Improvements</h2>
    <p>Several enhancements can be made to improve model performance. Implementing automated hyperparameter tuning, such as grid search or cyclic learning rates, can optimize training. Using Leaky ReLU instead of standard ReLU may help mitigate the dying ReLU problem. Additionally, exploring alternative loss functions like focal loss can be beneficial for handling class imbalances.</p>
    
 [For references](references.md)
</body>
</html>
