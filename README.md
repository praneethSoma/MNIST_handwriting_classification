
# MNIST Digit Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch. The code includes model definition, data preprocessing, training, validation, and testing.

## Project Structure

- `main.py`: Contains the main logic for loading data, training the model, evaluating performance, and visualizing results.
- `model.py`: Defines the CNN architecture.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

You can install the required packages using pip:

```bash
pip install torch torchvision matplotlib numpy
```

## Model Architecture

The CNN model defined in `model.py` consists of the following layers:

- Two convolutional layers:
  - `Conv2d`: 1 input channel, 32 output channels, kernel size of 3, stride of 1, padding of 1
  - `Conv2d`: 32 input channels, 64 output channels, kernel size of 3, stride of 1, padding of 1
- Max pooling layer:
  - `MaxPool2d`: kernel size of 2, stride of 2
- Fully connected layers:
  - `Linear`: input size of 7 * 7 * 64, output size of 128
  - `Linear`: input size of 128, output size of 10
- Dropout layer with a probability of 0.5
- ReLU activations after each convolutional and linear layer

## Data Preparation

The MNIST dataset is loaded and transformed to tensor format using `torchvision.datasets`. The training set is used for training the model, while the test set is split into validation and test sets:

```python
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

mnist_valset, mnist_testset = torch.utils.data.random_split(mnist_testset, [int(0.9 * len(mnist_testset)), int(0.1 * len(mnist_testset))])
```

Data loaders are created for batching and shuffling the datasets:

```python
train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(mnist_valset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=False)
```

## Training and Validation

The model is trained for 100 epochs using the Adam optimizer and CrossEntropyLoss:

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(no_epochs):
    ...
```

Training and validation loss are tracked and printed for each epoch. The model with the best validation loss is saved:

```python
if total_val_loss < best_val_loss:
    best_val_loss = total_val_loss
    torch.save(model.state_dict(), "model.dth")
```

## Testing

The saved model is loaded and evaluated on the test set to calculate the accuracy:

```python
model.load_state_dict(torch.load("model.dth"))
model.eval()

total = 0
for itr, (image, label) in enumerate(test_dataloader):
    ...
test_accuracy = total / (itr + 1)
print('Test accuracy {:.8f}'.format(test_accuracy))
```

## Results Visualization

The code visualizes some of the training images and the model's predictions on the test set using `matplotlib`:

```python
# visualize data
fig=plt.figure(figsize=(20, 10))
for i in range(1,3):
    img = transforms.ToPILImage(mode='L')(mnist_trainset[i][0])
    fig.add_subplot(1, 3, i)
    plt.title(mnist_trainset[i][1])
    plt.imshow(img)
plt.show()

# visualize results
fig=plt.figure(figsize=(20, 10))
for i in range(1, 11):
    img = transforms.ToPILImage(mode='L')(results[i][0].squeeze(0).detach().cpu())
    fig.add_subplot(2, 5, i)
    plt.title(results[i][1].item())
    plt.imshow(img)
plt.show()
```

## Usage

To run the project, simply execute the `main.py` script:

```bash
python main.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
