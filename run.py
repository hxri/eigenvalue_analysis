import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.nn.init as init

if __name__=="__main__":
    # Load the Iris dataset and preprocess it
    iris = load_iris()
    X = iris.data
    y = iris.target
    cols = np.shape(X)[1]

    # Hyperparameters
    input_size = 4
    hidden_size1 = 16
    hidden_size2 = 8
    hidden_size3 = 4
    num_classes = 3
    learning_rate = 0.01
    num_epochs = 200


    # Define the neural network model
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(hidden_size3, num_classes)
            
            # Initialize the weights to ones
            self.initialize_weights()
            
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
            out = self.relu3(out)
            out = self.fc4(out)
            return out

        def initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.constant_(m.weight, 0.1)
                    init.constant_(m.bias, 0.1)

    plot_data = []
    
    for itr in range(cols+1):
        if (itr == 0):
            # Convert to PyTorch tensors
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.int64)
            # print(X[:5, :])
        else:
            mu = np.mean(X[:, cols-itr].numpy()) + np.random.uniform(-2.5, 2.5)
            sigma = np.var(X[:, cols-itr].numpy()) + np.random.uniform(0, 2.5)
            X[:, cols-itr] = torch.tensor(np.random.normal(mu, sigma, (len(X), 1)).astype('float32')).squeeze()
            # print(X[:5, :])

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create the model
        model = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes)
        if(itr != 0):
            model.load_state_dict(torch.load('./{}_model.pth'.format(itr-1)))

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        gram_matrices = []

        ############################################
        # Training loop and model evaluation
        ############################################

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # # After training, you can access the weight matrices and print them
            with torch.no_grad():
                gm = []
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        print(f'Weight matrix for {name}:')
                        g = torch.mm(param.t(), param)
                        gm.append(g)
            gram_matrices.append(gm)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}\n\n')

        # Test the model
        with torch.no_grad():
            model.eval()
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            print(f'Accuracy on test set at iteration %d: {accuracy:.2f}' %itr)

        ##############################################################
        # Eigenvalue decomposition and mean and variance estimation
        ##############################################################

        evr = []
        for i in range(4):
            eigenvalue_variances = []
            # Calculate the eigenvalues for each Gram matrix
            for gram_matrix in gram_matrices:
                # Convert the Gram matrix to a NumPy array
                gram_matrix_np = gram_matrix[i].numpy()
                
                # Compute the eigenvalues using NumPy
                eigenvalues, _ = np.linalg.eig(gram_matrix_np)
                
                # Calculate the variance of the eigenvalues
                eigenvalue_variance = np.var(eigenvalues)
                
                eigenvalue_variances.append(eigenvalue_variance)

            # Calculate the mean and standard deviation of the eigenvalue variances
            mean_variance = np.mean(eigenvalue_variances)
            std_deviation = np.std(eigenvalue_variances)

            # print("Variance of Eigenvalues:")
            # for epoch, variance in enumerate(eigenvalue_variances):
            #     print(f'Epoch {epoch + 1}: {variance:.6f}')

            # print(f"Mean of Variance: {mean_variance:.6f}")
            # print(f"Standard Deviation: {std_deviation:.6f}")

            evr.append(eigenvalue_variances)

        plot_data.append(evr)
        torch.save(model.state_dict(), '{}_model.pth'.format(itr))

    np.save('plot_data.npy', np.array(plot_data))
    new = plot_data[0, :, :]
    for k in range(1, 5):
        new = np.hstack((new, plot_data[k, :, :]))
    # plt.plot(new[0], color='red')
    # plt.plot(new[1], color='green')
    # plt.plot(new[2], color='yellow')
    # plt.plot(new[3], color='blue')
