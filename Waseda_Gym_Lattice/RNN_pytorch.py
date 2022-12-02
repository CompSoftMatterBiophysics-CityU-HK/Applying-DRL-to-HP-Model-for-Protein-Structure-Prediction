"""
Example code of a simple RNN, GRU, LSTM on the MNIST dataset.

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-05-09 Initial coding

YouTube: https://www.youtube.com/watch?v=Gl2WXLIMvKA
"""

# Imports
import torch
# import torchvision  # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch import nn  # All neural network modules
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
# one row of the MNIST image is 28 pixels
# sequence_length = 28


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # input_size is the number of features for each time-step
        # we dont need to say explicitly how many sequences we want to have
        # RNN will work for any number of sequences we send (28 for MNIST)
        # batch_first use batch as the first axis --> (N, time_seq, features)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # hidden_size * sequence_length concatenates all the sequences from every hidden state
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        # x.size(0) is the number of mini-batches we send in at one time
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        # keep the batch as the 1st axis, and concatenate everything else
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Recurrent neural network with GRU (many-to-one)
class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # change basic RNN to GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Recurrent neural network with LSTM (many-to-one)
class RNN_LSTM_catAllHidden(nn.Module):
    """LSTM version that uses information from every hidden state"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM_catAllHidden, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # change basic RNN to LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # LSTM needs a separate cell state (LSTM needs both hidden and cell state)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        # need to give LSTM both hidden and cell state (h0, c0)
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


class RNN_LSTM_onlyLastHidden(nn.Module):
    """
    LSTM version that just uses the information from the last hidden state
    since the last hidden state has information from all previous states
    basis for BiDirectional LSTM
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM_onlyLastHidden, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # change basic RNN to LSTM
        # num_layers Default: 1
        # bias Default: True
        # batch_first Default: False
        # dropout Default: 0
        # bidirectional Default: False
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # remove the sequence_length
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Get data to cuda if possible
        x = x.to(device)
        # print("input x.size() = ", x.size())
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # LSTM needs a separate cell state (LSTM needs both hidden and cell state)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        # need to give LSTM both hidden and cell state (h0, c0)
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        # no need to reshape the out or concat
        # out is going to take all mini-batches at the same time + last layer + all features
        out = self.fc(out[:, -1, :])
        # print("forward out = ", out)
        return out

    def sample_action(self, obs, epsilon):
        # print("Sample Action called+++")
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            # print("coin < epsilon", coin, epsilon)
            # for 3actionStateEnv use [0,1,2]
            explore_action = random.randint(0,2)
            # print("explore_action = ", explore_action)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()

# Create a bidirectional LSTM
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # bidrectional=True for BiLSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        # hidden_size needs to expand both directions, *2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Get data to cuda if possible
        x = x.to(device)
        # print("input x.size() = ", x.size())
        # concat both directions, so need to times 2
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # the _ is the (hidden_state, cell_state), but not used
        out, _ = self.lstm(x, (h0, c0))
        # only take the last hidden state to send to the linear layer
        out = self.fc(out[:, -1, :])

        return out

    def sample_action(self, obs, epsilon):
        # print("Sample Action called+++")
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            # print("coin < epsilon", coin, epsilon)
            # for 3actionStateEnv use [0,1,2]
            explore_action = random.randint(0,2)
            # print("explore_action = ", explore_action)
            return explore_action
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()


# ******* only for debugging
if __name__ == "__main__":
    # execute only if run as a script
    print("running RNN_pytorch.py as script...")
    from torch import optim  # For optimizers like SGD, Adam, etc.
    import torchvision.datasets as datasets  # Standard datasets
    import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
    from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
    from tqdm import tqdm  # For a nice progress bar!
    from matplotlib import pyplot as plt

    # Nov 27 2021 compare FCN vs RNN speed
    from minimalRL_DQN import FCN_QNet
    from count_param_pytorch import count_parameters

    # MNIST Nx1x28x28
    input_size = 6 # lattice2D state output 4+2 (MINIST uses 28)
    # number of nodes in the hidden layers
    hidden_size = 256
    num_layers = 2 # 1 for BRNN
    num_classes = 10
    learning_rate = 0.005
    batch_size = 64
    num_epochs = 3

    # slice the MINIST into similar lattice2D state
    select_index = "[:,:,:,11:17]"  # 28//2 -3 to +3

    # Load Data
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # NOTE: each time the loader loads [batch_size, 1, img_w, img_h]
    sample_data, sample_targets = next(iter(train_loader))
    sample_data_width_6 = sample_data[:,:,:,11:17]
    print("sample_data_width_6.size() =")
    print(sample_data_width_6.size())
    print("sample_data_width_6.to(device=device).squeeze(1).size() =")
    print(sample_data_width_6.to(device=device).squeeze(1).size())

    print("sanity check for sample_targets.numpy()[0] = ", sample_targets.numpy()[0])
    plt.imshow(sample_data[0:1,:,:,:].numpy()[0][0])
    plt.show()
    # plot the width of 6 cropped sample_data
    plt.imshow(sample_data_width_6.numpy()[0][0])
    plt.show()

    print("sanity check for sample_targets.numpy()[-1] = ", sample_targets.numpy()[-1])
    plt.imshow(sample_data[-1:,:,:,:].numpy()[0][0])
    plt.show()
    # plot the width of 6 cropped sample_data
    plt.imshow(sample_data_width_6.numpy()[-1][0])
    plt.show()

    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    model = RNN_LSTM_onlyLastHidden(input_size, hidden_size, num_layers, num_classes).to(device)
    # or try with FCN_QNet
    # insize = input_size*28
    # print("insize = ", insize)
    # model = FCN_QNet(insize, num_classes).to(device)

    # display the model params
    count_parameters(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            # need to .squeeze(1) to remove the batch_sizex1x28x28's 1-axis
            data = data[:,:,:,11:17].to(device=device).squeeze(1)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent update step/adam step
            optimizer.step()

    # Check accuracy on training & test to see how good our model
    def check_accuracy(loader, model):
        """
        loader can be train_loader or test_loader
        """
        num_correct = 0
        num_samples = 0

        # Set model to eval
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                # need to do the .squeeze(1) for the check_accuracy too
                x = x[:,:,:,11:17].to(device=device).squeeze(1)
                y = y.to(device=device)

                # NOTE: scores is a 1x10 array for digits 0-9
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

        # Toggle model back to train
        model.train()
        return num_correct / num_samples


    print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
    print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

    """
    RNN inference playground
    """
    # NOTE: each time the loader loads [batch_size, 1, img_w, img_h]
    test_data, test_targets = next(iter(test_loader))

    # use a batch size of 4 for parallel inference
    start, end = 31, 35
    test_data_width_6 = test_data[start:end,:,:,11:17]
    print("test_data_width_6.size() =")
    print(test_data_width_6.size())
    print("test_data_width_6.to(device=device).squeeze(1).size() =")
    print(test_data_width_6.to(device=device).squeeze(1).size())

    print("sanity check for test_targets.numpy()[start] = ", test_targets.numpy()[start])
    # plot the width of 6 cropped sample_data
    plt.imshow(test_data_width_6.numpy()[0][0])
    plt.show()

    print("sanity check for test_targets.numpy()[end-1] = ", test_targets.numpy()[end-1])
    plt.imshow(test_data_width_6.numpy()[-1][0])
    plt.show()

    # run test data (batch size = 4) to the model prediction
    x = test_data_width_6.to(device=device).squeeze(1)
    print("x = ", x, x.size())
    y = test_targets[start:end]
    print("y = ", y, y.size())

    scores = model(x)
    print("scores = ", scores)
    _, predictions = scores.max(1)
    print('pred = ', predictions)

    ###############################
    # Now try with shortened input
    ###############################
    # use a batch size of 4 for parallel inference
    # start, end = 0, 4
    shortened_data = test_data[start:end,:,11:17,11:17]
    print("shortened_data.size() =")
    print(shortened_data.size())
    print("shortened_data.to(device=device).squeeze(1).size() =")
    print(shortened_data.to(device=device).squeeze(1).size())

    print("sanity check for test_targets.numpy()[start] = ", test_targets.numpy()[start])
    # plot the width of 6 cropped sample_data
    plt.imshow(shortened_data.numpy()[0][0])
    plt.show()

    print("sanity check for test_targets.numpy()[end-1] = ", test_targets.numpy()[end-1])
    plt.imshow(shortened_data.numpy()[-1][0])
    plt.show()

    # run one test data (batch size = 4) to the model prediction
    x = shortened_data.to(device=device).squeeze(1)
    print("x = ", x, x.size())
    y = test_targets[start:end]
    print("y = ", y, y.size())

    scores = model(x)
    print("scores = ", scores)
    _, predictions = scores.max(1)
    print('pred = ', predictions)

    #############################################
    # only crop by two rows
    # the predictions should be largely the same
    #############################################
    # use a batch size of 10 for parallel inference
    start, end = 31, 41
    shortened_data = test_data[start:end,:,1:-1,11:17]
    print("shortened_data.size() =")
    print(shortened_data.size())
    print("shortened_data.to(device=device).squeeze(1).size() =")
    print(shortened_data.to(device=device).squeeze(1).size())

    print("sanity check for test_targets.numpy()[start] = ", test_targets.numpy()[start])
    # plot the width of 6 cropped sample_data
    plt.imshow(shortened_data.numpy()[0][0])
    plt.show()

    print("sanity check for test_targets.numpy()[end-1] = ", test_targets.numpy()[end-1])
    plt.imshow(shortened_data.numpy()[-1][0])
    plt.show()

    # run one test data (batch size = 4) to the model prediction
    x = shortened_data.to(device=device).squeeze(1)
    print("x = ", x, x.size())
    y = test_targets[start:end]
    print("y = ", y, y.size())

    scores = model(x)
    print("scores = ", scores)
    _, predictions = scores.max(1)
    print('pred = ', predictions)

    # overall performance

    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            # need to do the .squeeze(1) for the check_accuracy too
            x = x[:,:,1:-1,11:17].to(device=device).squeeze(1)
            y = y.to(device=device)

            # NOTE: scores is a 1x10 array for digits 0-9
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print("x[:,:,1:-1,11:17] num_correct / num_samples = ",
        num_correct / num_samples)

    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            # need to do the .squeeze(1) for the check_accuracy too
            x = x[:,:,2:-2,11:17].to(device=device).squeeze(1)
            y = y.to(device=device)

            # NOTE: scores is a 1x10 array for digits 0-9
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print("x[:,:,2:-2,11:17] num_correct / num_samples = ",
        num_correct / num_samples) 


    ########################
    # Variable Length Batch
    ########################
    import numpy as np
    # 444444444444444444
    # first test digit 4
    manual_x_4 = np.array([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.6549, 0.2431],
         [0.0784, 0.0000, 0.0000, 0.0000, 0.5961, 0.9529],
         [0.9020, 0.0863, 0.0000, 0.0000, 0.4784, 0.9961],
         [0.9961, 0.3647, 0.0000, 0.0000, 0.3843, 0.9961],
         [0.7882, 0.0549, 0.0000, 0.0000, 0.1373, 0.9961],
         [0.0784, 0.0000, 0.0000, 0.0000, 0.0588, 0.8824],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7961],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7961],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7961],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7412],
         [0.8314, 0.8314, 0.6588, 0.4941, 0.4941, 0.7255],
         [0.9961, 0.9529, 0.9961, 0.9961, 0.9961, 0.9961],
         [0.2039, 0.1961, 0.4275, 0.5373, 0.5373, 0.7490],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4627],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4627],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4627],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5922],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7961],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7922],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1882],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]
    )
    print("manual_x_4, manual_x_4.shape = ", manual_x_4, manual_x_4.shape)
    plt.imshow(manual_x_4[0])
    plt.show()

    # run manual_x_4 (batch size = 1) to the model prediction
    x = torch.tensor(manual_x_4, device=device, dtype=torch.float)
    x = x.squeeze(1)
    print("x = ", x, x.size())
    print("y = 4")

    scores = model(x)
    print("scores = ", scores)
    _, predictions = scores.max(1)
    print('pred = ', predictions)

    # now a short 4!!
    short_4 = np.array([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #  [0.0000, 0.0000, 0.0000, 0.0000, 0.6549, 0.2431],
        #  [0.0784, 0.0000, 0.0000, 0.0000, 0.5961, 0.9529],
         [0.9020, 0.0863, 0.0000, 0.0000, 0.4784, 0.9961],
         [0.9961, 0.3647, 0.0000, 0.0000, 0.3843, 0.9961],
         [0.7882, 0.0549, 0.0000, 0.0000, 0.1373, 0.9961],
         [0.0784, 0.0000, 0.0000, 0.0000, 0.0588, 0.8824],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7961],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7961],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7961],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7412],
         [0.8314, 0.8314, 0.6588, 0.4941, 0.4941, 0.7255],
         [0.9961, 0.9529, 0.9961, 0.9961, 0.9961, 0.9961],
         [0.2039, 0.1961, 0.4275, 0.5373, 0.5373, 0.7490],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4627],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4627],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4627],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5922],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7961],
        #  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7922],
        #  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1882],
        #  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]
    )
    print("short_4, short_4.shape = ", short_4, short_4.shape)
    # size = [1, 18, 6]
    plt.imshow(short_4[0])
    plt.show()

    # run short_4 (batch size = 1) to the model prediction
    x = torch.tensor(short_4, device=device, dtype=torch.float)
    x = x.squeeze(1)
    print("x = ", x, x.size())
    print("y = 4")

    scores = model(x)
    print("scores = ", scores)
    _, predictions = scores.max(1)
    print('pred = ', predictions)

    # 6666666666666666
    # now test digit 6
    manual_x_6 = np.array([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.1647, 0.7529],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.3059, 0.9961],
         [0.0000, 0.0000, 0.0000, 0.1490, 0.6667, 0.9961],
         [0.0000, 0.0275, 0.5725, 0.9216, 0.9961, 0.8941],
         [0.1843, 0.8314, 0.9961, 0.9961, 0.9961, 0.2471],
         [0.4941, 0.9961, 0.9961, 0.9961, 0.7294, 0.0196],
         [0.9373, 0.9961, 0.9961, 0.8824, 0.1373, 0.0000],
         [0.9961, 0.9961, 0.9843, 0.4784, 0.0000, 0.0000],
         [0.9961, 0.9373, 0.3098, 0.0000, 0.0000, 0.0000],
         [0.9961, 0.9059, 0.0000, 0.0000, 0.0745, 0.2353],
         [0.9922, 0.4157, 0.0000, 0.3255, 0.8392, 0.9961],
         [0.8000, 0.0000, 0.0627, 0.8118, 0.9961, 1.0000],
         [0.5059, 0.0431, 0.5333, 0.9961, 0.9961, 0.9961],
         [0.5059, 0.5294, 0.9961, 0.9961, 0.9961, 0.7843],
         [0.5059, 0.0941, 0.9961, 1.0000, 0.8196, 0.0667],
         [0.5961, 0.0941, 0.9961, 0.9961, 0.6078, 0.0000],
         [0.9922, 0.5294, 0.9961, 0.9961, 0.2314, 0.0000],
         [0.9961, 0.9961, 0.9961, 0.9961, 0.8392, 0.4039],
         [0.9961, 0.9961, 0.9961, 0.9961, 0.9961, 0.9961],
         [0.2980, 0.5725, 0.9961, 0.9961, 0.9961, 0.8235],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]
    )
    print("manual_x_6, manual_x_6.shape = ", manual_x_6, manual_x_6.shape)
    plt.imshow(manual_x_6[0])
    plt.show()

    # run manual_x_6 (batch size = 1) to the model prediction
    x = torch.tensor(manual_x_6, device=device, dtype=torch.float)
    x = x.squeeze(1)
    print("x = ", x, x.size())
    print("y = 6")

    scores = model(x)
    print("scores = ", scores)
    _, predictions = scores.max(1)
    print('pred = ', predictions)

    # Now an elongated 6!
    long_6 = np.array([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.1843, 0.8314, 0.9961, 0.9961, 0.9961, 0.2471],
         [0.4941, 0.9961, 0.9961, 0.9961, 0.7294, 0.0196],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.1647, 0.7529],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.3059, 0.9961],
         [0.0000, 0.0000, 0.0000, 0.1490, 0.6667, 0.9961],
         [0.0000, 0.0275, 0.5725, 0.9216, 0.9961, 0.8941],
         [0.1843, 0.8314, 0.9961, 0.9961, 0.9961, 0.2471],
         [0.4941, 0.9961, 0.9961, 0.9961, 0.7294, 0.0196],
         [0.9373, 0.9961, 0.9961, 0.8824, 0.1373, 0.0000],
         [0.9961, 0.9961, 0.9843, 0.4784, 0.0000, 0.0000],
         [0.9961, 0.9373, 0.3098, 0.0000, 0.0000, 0.0000],
         [0.9961, 0.9059, 0.0000, 0.0000, 0.0745, 0.2353],
         [0.9922, 0.4157, 0.0000, 0.3255, 0.8392, 0.9961],
         [0.8000, 0.0000, 0.0627, 0.8118, 0.9961, 1.0000],
         [0.5059, 0.0431, 0.5333, 0.9961, 0.9961, 0.9961],
         [0.5059, 0.5294, 0.9961, 0.9961, 0.9961, 0.7843],
         [0.5059, 0.0941, 0.9961, 1.0000, 0.8196, 0.0667],
         [0.5961, 0.0941, 0.9961, 0.9961, 0.6078, 0.0000],
         [0.9922, 0.5294, 0.9961, 0.9961, 0.2314, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.1490, 0.6667, 0.9961],
         [0.9961, 0.9961, 0.9961, 0.9961, 0.8392, 0.4039],
         [0.9961, 0.9961, 0.9961, 0.9961, 0.9961, 0.9961],
         [0.2980, 0.5725, 0.9961, 0.9961, 0.9961, 0.8235],
         [0.8000, 0.0000, 0.0627, 0.8118, 0.9961, 1.0000],
         [0.5059, 0.0431, 0.5333, 0.9961, 0.9961, 0.9961],
         [0.5059, 0.5294, 0.9961, 0.9961, 0.9961, 0.7843],
         [0.5059, 0.0941, 0.9961, 1.0000, 0.8196, 0.0667],
         [0.5961, 0.0941, 0.9961, 0.9961, 0.6078, 0.0000],
         [0.9922, 0.5294, 0.9961, 0.9961, 0.2314, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.8000, 0.0000, 0.0627, 0.8118, 0.9961, 1.0000],
         [0.5059, 0.0431, 0.5333, 0.9961, 0.9961, 0.9961],
         [0.5059, 0.5294, 0.9961, 0.9961, 0.9961, 0.7843],
         [0.5059, 0.0941, 0.9961, 1.0000, 0.8196, 0.0667],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]
    )
    print("long_6, long_6.shape = ", long_6, long_6.shape)
    plt.imshow(long_6[0])
    plt.show()

    # run long_6 (batch size = 1) to the model prediction
    x = torch.tensor(long_6, device=device, dtype=torch.float)
    x = x.squeeze(1)
    print("x = ", x, x.size())
    print("y = 6")

    scores = model(x)
    print("scores = ", scores)
    _, predictions = scores.max(1)
    print('pred = ', predictions)

    # Combined batch size = 2, two short 4 and 6
    final_test = torch.zeros([2, 1, 26, 6], dtype=torch.float)
    final_test[0] = torch.tensor(manual_x_4[:, 1:-1, :], dtype=torch.float)
    final_test[1] = torch.tensor(manual_x_6[:, 1:-1, :], dtype=torch.float)

    # run one test data (batch size = 2) to the model prediction
    x = final_test.to(device=device).squeeze(1)
    print("x = ", x, x.size())
    print("y = ")

    scores = model(x)
    print("scores = ", scores)
    _, predictions = scores.max(1)
    print('pred = ', predictions)
