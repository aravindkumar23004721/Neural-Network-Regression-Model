# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Aravind Kumar SS
### Register Number:212223110004
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3=nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history={'loss':[]}
  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion= nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain,X_train,y_train,criterion,optimizer,epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    output=ai_brain(X_train)
    loss=criterion(output,y_train)
    loss.backward()
    optimizer.step()

    ai_brain.history['loss'].append(loss.item())
    if epoch%200==0:
      print(f'epoch {epoch} loss {loss.item():.6f}')
```
## Dataset Information

![Screenshot 2025-03-21 110227](https://github.com/user-attachments/assets/bc127b88-edce-42c6-a49d-35ab02eee25c)

## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-03-14 131715](https://github.com/user-attachments/assets/170f75a8-38ec-4b84-95ab-ec8a45f1cd24)

### New Sample Data Prediction

![Screenshot 2025-03-21 105652](https://github.com/user-attachments/assets/0a401cfe-24ab-4ec0-ac88-5cb873ce4bf7)

## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
