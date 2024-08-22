import os  # module for os operating, files
import random  # used for randomness in the code
import torchaudio  # deals with audio data
import torchaudio.transforms as T  # for applying audio tranformations
from sklearn.model_selection import train_test_split  # splits data

import torch  # core pytorch library
import torch.optim as optim  # used for updating model weights during training, optimisation algorithms
import torch.nn as nn  # classes and tools for neural networks
import torch.nn.functional as F  # function for operation of nn like relu, sigmoid
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # load data in batches for traning and creates datasets
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    log_loss,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_audio_folder(folder_path, label, limit, target_sr=8000):
    # Initialize lists to store audio samples and labels
    audio_samples = []
    labels = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            # Load the audio file using torchaudio
            file_path = os.path.join(folder_path, filename)

            waveform, original_sample_rate = torchaudio.load(
                file_path
            )  # loads audio data in waveform and sample rate
            resampler = T.Resample(
                orig_freq=original_sample_rate, new_freq=target_sr
            )  # used to convert all sample rates from orig to 8000
            waveform_resampled = resampler(waveform)  # standardize all to 8000

            # Ensure waveform is exactly 8000 samples long
            if waveform_resampled.shape[1] > 8000:
                waveform_resampled = waveform_resampled[
                    :, :8000
                ]  # change to 8000 again if not
            elif waveform_resampled.shape[1] < 8000:
                padding_size = 8000 - waveform_resampled.shape[1]
                waveform_resampled = F.pad(
                    waveform_resampled, (0, padding_size)
                )  # if its shorter it fills with 0 to reach 8000

            # Add the tensor to the list of samples
            audio_samples.append(waveform_resampled)

            # Add the corresponding label
            # Convert the label to a one-hot encoded tensor
            probabilities = [0.0 for i in range(10)]
            probabilities[label] = 1.0
            label_tensor = torch.tensor(
                probabilities, dtype=torch.float32
            )  # Ensure correct tensor creation

            # Add the one-hot encoded label tensor to the list
            labels.append(label_tensor)

            if len(audio_samples) == limit:
                break

    # Stack the samples and labels into tensors
    audio_samples_tensor = torch.stack(
        audio_samples
    )  # Shape: (n_samples, n_mel_bins, n_time_steps)
    labels_tensor = torch.stack(labels)  # Shape: (n_samples,)

    return audio_samples_tensor, labels_tensor


def to_pairs(path, limit):
    digit_limit = limit // 10
    names = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]

    audio_samples_tensors = []
    labels_tensors = []

    for index, name in enumerate(names):
        audio_samples_tensor, labels_tensor = process_audio_folder(
            f"{path}/{name}", label=index, limit=digit_limit
        )
        audio_samples_tensors.append(audio_samples_tensor)
        labels_tensors.append(labels_tensor)

    audio_samples_tensor = torch.cat(audio_samples_tensors, dim=0)
    labels_tensor = torch.cat(labels_tensors, dim=0)
    return audio_samples_tensor, labels_tensor


class NumbersDataset(Dataset):  # dataset class for audio data
    def __init__(self, audio_samples_tensor, labels_tensor):  # initiate, intake pairs
        self.audio_samples_tensor = audio_samples_tensor  # store pairs
        self.labels_tensor = labels_tensor  # store pairs

    def __len__(self):  # length method
        return len(self.audio_samples_tensor)  # returns number of pairs

    def __getitem__(self, idx):  # new method
        audio_samples_tensor = self.audio_samples_tensor[idx]  # assign variables
        labels_tensor = self.labels_tensor[idx]  # assign variables
        # [1, 8000], [10]
        return (
            audio_samples_tensor,
            labels_tensor,
        )  # returns the pair of spectrograms and label as a tensor


class CnnBlock(nn.Module):  # make CNN block class
    def __init__(
        self, in_channels, out_channels, kernel=3, stride=1, padding=1
    ):  # initialize with hyperparametrs
        super(CnnBlock, self).__init__()  # call the parent class constructor - module
        self.conv1 = nn.Conv1d(  # first convolutional layer with parameters
            in_channels,  # ToDo: read about convolution more
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(
            out_channels
        )  # batch normalization after first convolution
        self.conv2 = nn.Conv1d(  # second convolutional layer
            out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(
            out_channels
        )  # batch normalization after second convolution

        # Residual connection
        # ToDo: explain why it is needed for CNN (hint: signal decay)
        self.shortcut = (
            nn.Sequential()
        )  # make an empty sequential layer for the shortcut connection
        if (
            stride != 1 or in_channels != out_channels
        ):  # if stride or channel count changes, adjust the shortcut with a 1x1 convolution and batch norm
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        self.relu = nn.ReLU()  # relu activated

    def forward(self, x):
        out = F.relu(
            self.bn1(self.conv1(x))
        )  # apply the first conv layer, batch norm, and ReLU activation
        out = self.bn2(self.conv2(out))  # apply the second conv layer and batch norm
        out = out + self.shortcut(x)  # add the shortcut to the output
        out = F.relu(out)  # activate relu to output
        return out


class NumbersModel(nn.Module):  # new class
    def __init__(self):
        super(
            NumbersModel, self
        ).__init__()  # call the parent class constructor - voicematchmodule
        # [B, 1, 8000]
        self.cnn_layers = nn.Sequential(  # define CNN layers
            CnnBlock(1, 16),  # first cnn block [B, 16, 8000]
            CnnBlock(16, 32, stride=2),  # second cnn block [B, 32, 4000]
            CnnBlock(32, 64, stride=2),  # third cnn block [B, 64, 2000]
            CnnBlock(64, 128, stride=2),  # fourth cnn block [B, 128, 1000]
            CnnBlock(128, 256, stride=2),  # fourth cnn block [B, 256, 500]
        )

        # [B, 128 * 1000] -  incoming shape
        self.fc_layers = nn.Sequential(  # define fully connected layers
            nn.Linear(256 * 500, 256),  # first linear layer
            nn.ReLU(),  # relu activation
            nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
            nn.Linear(256, 128),  # second linera layer
            nn.ReLU(),  # relu activation
            nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
            nn.Linear(128, 64),  # third linera layer
            nn.ReLU(),  # relu activation
            nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
            nn.Linear(64, 10),  # fourth linera layer
            # nn.Softmax(), #softmax activation
        )
        """
        X:          [0,     1,      2,      3,      4,      5,      6,      7,       8,      9]
        Y= P(X):    [0.1,   0.1,    0.1,    0.2,    0.2,    0.1,    0.0,     0.0,    0.1,    0.1]
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 8000]

        # Forward through CNN layers
        x = self.cnn_layers(x)  # pass the input through the CNN layers
        # x: [B, 128, 1000]

        # flatten x1
        x = x.view(x.size(0), -1)
        # x: [B, 128 * 1000]

        # Forward through linear (fully-connected) layers
        x = self.fc_layers(
            x
        )  # pass through the fully connected layers to get the final output
        # x: [B, 10]
        return x  # return the final output (probability distribution across digits)


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):  # create function for training
    model.train()  # set the model to training mode
    with torch.autograd.set_detect_anomaly(True):  # enable anomaly detection for debugging
        for epoch in range(num_epochs):  # loop over the number of epochs
            running_loss = 0.0  # set running loss for the epoch
            for (audio,labels,) in train_loader:  # batch and iterate over data in the training set
                # audio: [B, 1, 8000]
                # labels: [B, 10]

                optimizer.zero_grad()  # zero the gradients before each forward pass

                # Move data to the appropriate device
                audio = audio.to(DEVICE)
                labels = labels.to(DEVICE)

                # Forward pass through the model
                # audio: [B, 1, 8000]
                # labels: [B, 10]
                outputs = model(audio)
                # outputs: [B, 10]

                loss = criterion(outputs, labels)  # calculate the loss
                # [B, 10]

                # Backward pass and optimize
                loss.backward()  # compute gradients for backpropagation
                optimizer.step()  # update the model parameters

                running_loss += loss.item() * audio.size(0)  # calculate running loss

            epoch_loss = running_loss / len(
                train_loader.dataset
            )  # calculate average loss for epoch

            # Evaluate on test data
            model.eval()  # Set model to evaluation mode
            test_loss = 0.0  # initialize test loss
            all_outputs = []  # list to store all predictions
            all_labels = []  # list to store all true labels
            with torch.no_grad():  # disable gradient computation for testing
                for audio, labels in test_loader:  # iterate over the test data
                    # audio: [B, 1, 8000]
                    # labels: [B, 10]

                    audio = audio.to(DEVICE)
                    labels = labels.to(DEVICE)

                    outputs = model(audio)  # forward pass and remove single dimension
                    # outputs: [B, 10]
                    loss = criterion(outputs, labels)  # calculate the test loss
                    # [B, 10]

                    test_loss += loss.item() * audio.size(0)  # calculate test loss

                    all_outputs.append(F.softmax(outputs, dim=-1))  # store predictions
                    all_labels.append(labels)  # store true labels

            test_loss /= len(test_loader.dataset)  # calculate average test loss
            all_outputs = torch.cat(
                all_outputs
            )  # put all predictions into a single tensor
            all_labels = torch.cat(
                all_labels
            )  # put all true labels into a single tensor
            accuracy, f1, conf_matrix = evaluate(
                all_outputs, all_labels
            )  # evaluate precision, recall, and F1 score
            print(  # print the metrics for the current epoch
                f"Epoch [{epoch+1}/{num_epochs}]. Train loss: {epoch_loss:.4f}. Test Loss: {test_loss:.4f}. "
                f"Accuracy {accuracy:.4f}. F1: {f1:.4f}."
            )
            model.train()  # Set model back to train mode


def evaluate(predictions_distribution, labels_distribution):
    # predictions_distribution: [B, 10] # class per digit
    # labels_distribution: [B, 10] # class per digit

    # Convert probabilities to predicted class labels
    predictions = torch.argmax(predictions_distribution, dim=1)
    labels = torch.argmax(labels_distribution, dim=1)

    # Convert to numpy arrays for sklearn functions
    predictions_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Accuracy
    accuracy = accuracy_score(labels_np, predictions_np)

    # Confusion Matrix
    conf_matrix = confusion_matrix(labels_np, predictions_np)

    # Precision, Recall, F1 Score
    _, _, f1, _ = precision_recall_fscore_support(
        labels_np, predictions_np, average="weighted"
    )

    # Return results as a dictionary
    return accuracy, f1, conf_matrix


def run_test(dataset_limit: int, batch_size: int, lr: int):
    # Prepare data
    print("Preparing data")
    # set a limit on the total number of pairs to be generated
    # one = read_as_dict("/root/learning-nn/resources/speech_commands/one", 1000)

    # load data into a dict from marvin folder
    audio_samples_tensor, labels_tensor = to_pairs(
        "/root/learning-nn/resources/speech_commands", dataset_limit
    )
    print(f"Read {len(audio_samples_tensor)} pairs")

    as_train, as_test, l_train, l_test = train_test_split(
        audio_samples_tensor, labels_tensor, test_size=0.2, random_state=42
    )

    # Setting up the training
    print(
        f"Running training. dataset_limit={dataset_limit}. batch_size={batch_size}. lr={lr}"
    )
    model = NumbersModel().to(DEVICE)
    train_loader = DataLoader(
        dataset=NumbersDataset(as_train, l_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=NumbersDataset(as_test, l_test), batch_size=batch_size, shuffle=True
    )

    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50)

    print("Finished")


"""
dataset_limit = 10_000  batch_size = 64    lr = 0.0005          Epoch [48/50]. Train loss: 0.5297. Test Loss: 1.3183. Accuracy 0.6765. F1: 0.6769.

dataset_limit = 20_000  batch_size = 16    lr = 0.0005         *Epoch [16/50]. Train loss: 0.6332. Test Loss: 0.9068. Accuracy 0.7007. F1: 0.7027.
dataset_limit = 20_000  batch_size = 32    lr = 0.0005          Epoch [16/50]. Train loss: 0.4637. Test Loss: 0.7815. Accuracy 0.7927. F1: 0.7931.
dataset_limit = 20_000  batch_size = 64    lr = 0.0005          Epoch [26/50]. Train loss: 0.5147. Test Loss: 0.8011. Accuracy 0.7688. F1: 0.7694.

dataset_limit = 20_000  batch_size = 256   lr = 0.0005          Epoch [35/50]. Train loss: 0.7958. Test Loss: 1.1784. Accuracy 0.6048. F1: 0.5849. 
"""


for dataset_limit in [20_000]:
    for batch_size in [32, 64]:
        for lr in [0.001, 0.0005, 0.0002, 0.0001]:
            run_test(dataset_limit, batch_size, lr)

"""
Log:

Aug 12th:
- Achieved accuracy of 0.25 (25%) from the first try. Test loss is stuck.
    Implementing residual connections for FC layers to improve regularization.

Aug 14th

Epoch [31/50]. Train loss: 1.2276. Test Loss: 1.5020. Accuracy 0.4010. F1: 0.3684.

lr = 0.0005
10000

Epoch [35/50]. Train loss: 0.7052. Test Loss: 1.3897. Accuracy 0.5670. F1: 0.5394.
Epoch [36/50]. Train loss: 0.7009. Test Loss: 2.4654. Accuracy 0.3365. F1: 0.3295.
Epoch [37/50]. Train loss: 0.6739. Test Loss: 1.7441. Accuracy 0.5000. F1: 0.4665.
Epoch [38/50]. Train loss: 0.6382. Test Loss: 1.3646. Accuracy 0.6330. F1: 0.6236.
Epoch [39/50]. Train loss: 0.6166. Test Loss: 1.5262. Accuracy 0.6105. F1: 0.5992.
Epoch [40/50]. Train loss: 0.6386. Test Loss: 1.6011. Accuracy 0.6020. F1: 0.5972.
Epoch [41/50]. Train loss: 0.6199. Test Loss: 1.3535. Accuracy 0.6595. F1: 0.6582.
Epoch [42/50]. Train loss: 0.5729. Test Loss: 1.4685. Accuracy 0.6425. F1: 0.6406.
Epoch [43/50]. Train loss: 0.5782. Test Loss: 1.4868. Accuracy 0.6565. F1: 0.6545.
Epoch [44/50]. Train loss: 0.5574. Test Loss: 1.8198. Accuracy 0.5400. F1: 0.5225.
Epoch [45/50]. Train loss: 0.5561. Test Loss: 1.9693. Accuracy 0.4660. F1: 0.4523.
Epoch [46/50]. Train loss: 0.5466. Test Loss: 1.5575. Accuracy 0.6130. F1: 0.6077.
Epoch [47/50]. Train loss: 0.5308. Test Loss: 2.0990. Accuracy 0.5285. F1: 0.5179.
Epoch [48/50]. Train loss: 0.5297. Test Loss: 1.3183. Accuracy 0.6765. F1: 0.6769.
Epoch [49/50]. Train loss: 0.5054. Test Loss: 1.5252. Accuracy 0.6180. F1: 0.6182.
Epoch [50/50]. Train loss: 0.5236. Test Loss: 1.6018. Accuracy 0.5910. F1: 0.5889.

"""
