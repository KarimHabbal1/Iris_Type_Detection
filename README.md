Iris Flower Classification with PyTorch
This project implements a neural network model using PyTorch to classify iris flowers into three species: Setosa, Versicolor, and Virginica. The model is trained on the classic Iris dataset, which consists of 150 samples with four features each: sepal length, sepal width, petal length, and petal width.

Dataset
The dataset is a well-known benchmark in machine learning and can be found here. The species column represents the target labels and is encoded as follows:

Setosa: 0.0
Versicolor: 1.0
Virginica: 2.0
Model Architecture
The model is a simple feed-forward neural network with:

Input layer of size 4 (for the four features)
Two hidden layers with 10 and 6 units, respectively, and ReLU activations
Output layer of size 3 (corresponding to the three flower species)
Neural Network Structure

Training
Loss Function: CrossEntropyLoss (suitable for multi-class classification).
Optimizer: Adam with a learning rate of 0.01.
Epochs: 100
The training loop runs for 100 epochs, recording and printing the loss every 10 epochs.

Requirements
pandas
torch
matplotlib
scikit-learn
Usage
Clone the repository or download the script.
Install the required libraries:
bash
Copy code
pip install pandas torch matplotlib scikit-learn
Run the script to train and test the model:
bash
Copy code
python iris_classification.py
Code Explanation
Data Preprocessing: The dataset is loaded and the species column is encoded numerically. The dataset is then split into training and testing sets.
Model Definition: The Model class defines the architecture of the neural network.
Training Loop: The model is trained for 100 epochs, and the loss is tracked and printed.
Evaluation: After training, the model is evaluated on the test data.
Results
The final evaluation loss on the test data is printed. You can further explore model accuracy or add additional metrics as needed.
