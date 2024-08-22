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
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_audio_folder(path, limit):  # create new function with two arguments
    result = {}  # empty dictionary

    for filename in os.listdir(
        path
    ):  # start loop for all files in the path, lists all files
        if filename == ".DS_Store":  # skips system file
            continue
        file_path = os.path.join(
            path, filename
        )  # creates new var of path and name of the file
        # print(f"Processing file: {file_path}") #used for checking the process

        waveform, original_sample_rate = torchaudio.load(
            file_path
        )  # loads audio data in waveform and sample rate
        resampler = T.Resample(
            orig_freq=original_sample_rate, new_freq=8000
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

        username = filename.split("_")[0]  # extract userid from file name
        if username in result:
            result[username].append(
                waveform_resampled
            )  # if user exist in dictionary, add file to it
        else:
            result[username] = [waveform_resampled]  # add user and file into dict
        if len(result) >= limit:  # stop if dict hits the limit per user
            break

    positive_triples = to_positive_triples(result, limit=int(limit * 0.5))
    negative_triples = to_negative_triples(result, limit=int(limit * 0.5))
    return positive_triples, negative_triples


def to_positive_triples(
    user_to_audio, limit
):  # new function that takes dict and limit number of rows for matching samples
    # construct queue of user samples
    triples = []  # empty list to store all pairs
    for user, audios in user_to_audio.items():  # loop through each user's audio samples
        user_triples = []  # list to hold pairs for current user in the loop
        for i in range(len(audios)):  # loop through each audio sample
            for j in range(i + 1, len(audios)):  # loop thorugh next audio samples
                if len(user_triples) < 5:  # set limit to 5 pairs per users
                    user_triples.append(
                        (audios[i], audios[j], torch.tensor([1.0], dtype=torch.float32))
                    )  # add to list and put label 1
                else:  # ?if list is full
                    break  # ?break the cycle
            if len(user_triples) >= limit:  # ?
                break
        triples.extend(user_triples)  # add user pairs to the main list

    random.Random(42).shuffle(triples)  # randomly shuffle all paris

    return triples[:limit]  # ?return limited number of pairs


def to_negative_triples(user_to_audio, limit):  # set fnction to create negative pairs
    neg_triples = []  # empty list to sore

    # construct queue of user samples
    index_user_sample = []  # to hold user pairs
    for user, audios in user_to_audio.items():  # loop through users
        for audio in audios:  # loop through audios
            index_user_sample.append((user, audio))  # store pairs in the list

    queue1 = list(index_user_sample)  # copy user audio list
    random.Random(42).shuffle(queue1)  # randomly shuffle new list

    queue2 = list(index_user_sample)  # another copy of user audio list
    random.Random(24).shuffle(queue2)  # randomly shuffle new list

    for i in range(len(queue1)):  # loop through queue lists
        if queue1[i][0] != queue2[i][0]:  # check if users are same
            neg_triples.append(
                (queue1[i][1], queue2[i][1], torch.tensor([0.0], dtype=torch.float32))
            )  # add a pair if they are not same and set label 0

            if len(neg_triples) == limit:  # break if list hit limit
                break

    return neg_triples  # return list with negative pairs


def to_triples(path, limit):
    digit_limit = limit // 10
    digits = [
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

    all_positive_triples = []
    all_negative_triples = []

    for digit in digits:
        positive_triples, negative_triples = process_audio_folder(
            f"{path}/{digit}", limit=digit_limit
        )
        all_positive_triples.extend(positive_triples)
        all_negative_triples.extend(negative_triples)

    return all_positive_triples, all_negative_triples


def evaluate(predictions, labels):
    true_positives = (
        torch.logical_and(predictions >= 0.5, labels == 1).sum().item()
    )  # calculate true positives
    false_positives = (
        torch.logical_and(predictions >= 0.5, labels == 0).sum().item()
    )  # calculate false positives
    true_negatives = (
        torch.logical_and(predictions < 0.5, labels == 0).sum().item()
    )  # calculate true negatives
    false_negatives = (
        torch.logical_and(predictions < 0.5, labels == 1).sum().item()
    )  # calculate false negatives

    # calculate precision: the ratio of true positives to the total predicted positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )

    # calculate recall: the ratio of true positives to the total actual positives
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0
    )

    # calculate F1 score: the harmonic mean of precision and recall
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )

    # calculate accuracy: the ratio of correct predictions (true positives + true negatives) to the total predictions
    accuracy = (
        (true_positives + true_negatives)
        / (true_positives + false_positives + true_negatives + false_negatives)
        if true_positives + false_positives + true_negatives + false_negatives > 0
        else 0
    )

    # return results
    return precision, recall, f1_score, accuracy


def load_data(limit: int, cache_dir: str = None):
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"data_limit_{limit}.pkl")

    if cache_file and os.path.exists(cache_file):
        print(f"Loading data from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            train, test = pickle.load(f)
    else:
        # Prepare data
        print("Preparing data")
        positive_triples, negative_triples = to_triples(
            "/Users/Dolg/Documents/Flatiron/capstone/data_numbers_wav", limit
        )

        print(f"Read {len(positive_triples)} positive_pairs")
        print(f"Read {len(negative_triples)} negative_pairs")

        p_train, p_test = train_test_split(
            positive_triples, test_size=0.2, random_state=42
        )
        n_train, n_test = train_test_split(
            negative_triples, test_size=0.2, random_state=42
        )

        train = p_train + n_train
        test = p_test + n_test
        random.Random(42).shuffle(train)
        random.Random(42).shuffle(test)

        if cache_file:
            print(f"Saving data to cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump((train, test), f)

    return train, test

class VoiceDataset(Dataset):
    def __init__(self, list_triple):
        self.list_triple = list_triple

    def __getitem__(self, index):
        return self.list_triple[index]
    
    def __len__(self):
        return len(self.list_triple)

class BlockCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.batchnorm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.batchnorm2 = nn.BatchNorm1d(out_channels)

        self.shortcut_batchnorm = nn.BatchNorm1d(out_channels)
        self.shortcut_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        self.relu2 = nn.ReLU()

        self.maxpooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, audio):
        x = self.conv1(audio)
        x = self.batchnorm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)

        x = x + self.shortcut_batchnorm(self.shortcut_conv(audio))

        x = self.relu2(x)
        x = self.maxpooling(x)

        return x
    


"""
CNN Block Architecture

[B, 1, 8000] (in_channels, out_channels, kernel=3, stride=1, padding=1)

(1, 4, stride=5)

1. Conv1 (in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding) [B, 4, 8000]
2. BatchNorm1 (out_channels) [B, 4, 8000]
3. Relu1 [B, 4, 8000]

4. Conv2 (out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding) [B, 4, 8000]
5. BatchNorm2 (out_channels) [B, 4, 8000]

6. x + shortcut(audio) [B, 4, 8000] + [B, 4, 8000] = [B, 4, 8000]
    Conv1(in_channels, out_channels, kernel_size=1, stride=1) [B, 4, 8000]
    BatchNorm(out_channels) [B, 4, 8000]

7. ReLu [B, 4, 8000]
8. Pooling MaxPool1d(kernel_size=kernel, stride=stride, padding=padding) [B, 4, 1600]
"""
class VoiceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = BlockCNN(in_channels=1, out_channels=4, stride=5)
        self.cnn2 = BlockCNN(in_channels=4, out_channels=8, stride=5)
        self.cnn3 = BlockCNN(in_channels=8, out_channels=16, stride=5)
        self.cnn4 = BlockCNN(in_channels=16, out_channels=32, stride=4)

        self.fc1 = nn.Linear(in_features=32 * 16 * 2, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio1, audio2):
        x1 = self.cnn1(audio1)
        x1 = self.cnn2(x1)
        x1 = self.cnn3(x1)
        x1 = self.cnn4(x1)

        x2 = self.cnn1(audio2)
        x2 = self.cnn2(x2)
        x2 = self.cnn3(x2)
        x2 = self.cnn4(x2)

        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)

        x = torch.cat((x1, x2), 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

"""
Input: audio1 [B, 1, 8000], audio2 [B, 1, 8000]

1. CNN Block1 (1, 4, stride=5) > [B, 4, 1600]
2. CNN Block2 (4, 8, stride=5) > [B, 8, 320]
3. CNN Block3 (8, 16, stride=5) > [B, 16, 64]
4. CNN Block 4 (16, 32, stride=4) > [B, 32, 16]

5. Flatten both tensors, concat [B, 32 * 16 * 2]

6. FC layer (32 * 16 * 2, 16) [B, 16]
7. Relu [B, 16]
8. Dropout (p=0.5) [B, 16]
9. FC Layer (16, 1) [B, 1]
10. Sigmoid [B, 1]
"""

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0
        for audio1, audio2, label in train_loader:
            optimizer.zero_grad()

            audio1.to(DEVICE)
            audio2.to(DEVICE)
            label.to(DEVICE)

            prediction = model(audio1, audio2)
            loss = criterion(prediction, label)
            loss.backward()

            optimizer.step()

            running_loss += loss * audio1.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()

        predictions = []
        labels = []

        running_loss = 0
        with torch.no_grad():
            for audio1, audio2, label in test_loader:
                audio1 = audio1.to(DEVICE)
                audio2 = audio2.to(DEVICE)
                label = label.to(DEVICE)

                prediction = model(audio1, audio2)
                loss = criterion(prediction, label) 
    
                running_loss += loss * audio1.size(dim=0)

                predictions.append(prediction)
                labels.append(label)

        test_loss = running_loss / len(test_loader.dataset)

        predictions = torch.cat(predictions) 
        labels = torch.cat(labels) 
        precision, recall, f1_score, accuracy = evaluate(predictions, labels)


        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

        model.train()

def run_test(
    train,
    test,
    dataset_limit: int,
    batch_size: int,
    lr: int,
    epochs: int,
    weight_decay: float,
):
    # Setting up the training
    print(
        f"Running training. dataset_limit={dataset_limit}. batch_size={batch_size}. lr={lr}. epochs={epochs}. weight_decay={weight_decay}"
    )
    model = VoiceModel().to(DEVICE)
    train_loader = DataLoader(
        dataset=VoiceDataset(train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=VoiceDataset(test), batch_size=batch_size, shuffle=True
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train the model
    train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=epochs
    )

    print("Finished")


# train, test = load_data(20_000)
# run_test(train, test, 20_000, 64, 0.0005)

for epochs in [50]:
    for dataset_limit in [30_000]:
        train, test = load_data(
            dataset_limit, cache_dir="/Users/Dolg/Documents/cache"
        )
        for lr in [0.001, 0.0005, 0.0002, 0.0001]:
            for weight_decay in [0.001, 0.0005]:
                for batch_size in [64, 128]:
                    run_test(
                        train, test, dataset_limit, batch_size, lr, epochs, weight_decay
                    )

"""

########################################
########################################
########################################

Test 11
Running training. dataset_limit=20000. batch_size=128. lr=0.001. epochs=50. weight_decay=0.0005
Epoch: 1/50, Train Loss: 0.6573, Test Loss: 0.5998, Accuracy: 0.7170
Epoch: 2/50, Train Loss: 0.6296, Test Loss: 0.5799, Accuracy: 0.7312
Epoch: 3/50, Train Loss: 0.6166, Test Loss: 0.5717, Accuracy: 0.7338
Epoch: 4/50, Train Loss: 0.6145, Test Loss: 0.5743, Accuracy: 0.7372
Epoch: 5/50, Train Loss: 0.6105, Test Loss: 0.5618, Accuracy: 0.7460
Epoch: 6/50, Train Loss: 0.6013, Test Loss: 0.5491, Accuracy: 0.7560
Epoch: 7/50, Train Loss: 0.5970, Test Loss: 0.5479, Accuracy: 0.7475
Epoch: 8/50, Train Loss: 0.5945, Test Loss: 0.5427, Accuracy: 0.7538
Epoch: 9/50, Train Loss: 0.5787, Test Loss: 0.5204, Accuracy: 0.7538
Epoch: 10/50, Train Loss: 0.5769, Test Loss: 0.5169, Accuracy: 0.7712
Epoch: 11/50, Train Loss: 0.5623, Test Loss: 0.4909, Accuracy: 0.7915
Epoch: 12/50, Train Loss: 0.5533, Test Loss: 0.4731, Accuracy: 0.7993
Epoch: 13/50, Train Loss: 0.5459, Test Loss: 0.4613, Accuracy: 0.8117
Epoch: 14/50, Train Loss: 0.5301, Test Loss: 0.4465, Accuracy: 0.8190
Epoch: 15/50, Train Loss: 0.5284, Test Loss: 0.6438, Accuracy: 0.6707
Epoch: 16/50, Train Loss: 0.5191, Test Loss: 0.4263, Accuracy: 0.8275
Epoch: 17/50, Train Loss: 0.5113, Test Loss: 0.4276, Accuracy: 0.8275
Epoch: 18/50, Train Loss: 0.5009, Test Loss: 0.5715, Accuracy: 0.6943
Epoch: 19/50, Train Loss: 0.4935, Test Loss: 0.4032, Accuracy: 0.8417
Epoch: 20/50, Train Loss: 0.4844, Test Loss: 0.4529, Accuracy: 0.7863
Epoch: 21/50, Train Loss: 0.4771, Test Loss: 0.3600, Accuracy: 0.8648
Epoch: 22/50, Train Loss: 0.4770, Test Loss: 0.3601, Accuracy: 0.8650
Epoch: 23/50, Train Loss: 0.4738, Test Loss: 0.3601, Accuracy: 0.8622
Epoch: 24/50, Train Loss: 0.4694, Test Loss: 0.3485, Accuracy: 0.8590
Epoch: 25/50, Train Loss: 0.4615, Test Loss: 0.3693, Accuracy: 0.8535
Epoch: 26/50, Train Loss: 0.4593, Test Loss: 0.3717, Accuracy: 0.8440
Epoch: 27/50, Train Loss: 0.4524, Test Loss: 0.3356, Accuracy: 0.8738
Epoch: 28/50, Train Loss: 0.4586, Test Loss: 0.3423, Accuracy: 0.8612
Epoch: 29/50, Train Loss: 0.4473, Test Loss: 0.3423, Accuracy: 0.8648
Epoch: 30/50, Train Loss: 0.4432, Test Loss: 0.3515, Accuracy: 0.8545
Epoch: 31/50, Train Loss: 0.4430, Test Loss: 0.3432, Accuracy: 0.8608
Epoch: 32/50, Train Loss: 0.4421, Test Loss: 0.3433, Accuracy: 0.8610
Epoch: 33/50, Train Loss: 0.4377, Test Loss: 0.3230, Accuracy: 0.8715
Epoch: 34/50, Train Loss: 0.4387, Test Loss: 0.3400, Accuracy: 0.8648
Epoch: 35/50, Train Loss: 0.4370, Test Loss: 0.3246, Accuracy: 0.8685
Epoch: 36/50, Train Loss: 0.4318, Test Loss: 0.3656, Accuracy: 0.8433
Epoch: 37/50, Train Loss: 0.4244, Test Loss: 0.3256, Accuracy: 0.8640
Epoch: 38/50, Train Loss: 0.4306, Test Loss: 0.3337, Accuracy: 0.8610
Epoch: 39/50, Train Loss: 0.4229, Test Loss: 0.3276, Accuracy: 0.8670
Epoch: 40/50, Train Loss: 0.4165, Test Loss: 0.3257, Accuracy: 0.8655
Epoch: 41/50, Train Loss: 0.4193, Test Loss: 0.3410, Accuracy: 0.8615
Epoch: 42/50, Train Loss: 0.4184, Test Loss: 0.3556, Accuracy: 0.8472
Epoch: 43/50, Train Loss: 0.4098, Test Loss: 0.3416, Accuracy: 0.8572
Epoch: 44/50, Train Loss: 0.4181, Test Loss: 0.3332, Accuracy: 0.8620
Epoch: 45/50, Train Loss: 0.4100, Test Loss: 0.3280, Accuracy: 0.8642
Epoch: 46/50, Train Loss: 0.4093, Test Loss: 0.3190, Accuracy: 0.8685
Epoch: 47/50, Train Loss: 0.4119, Test Loss: 0.3406, Accuracy: 0.8608
Epoch: 48/50, Train Loss: 0.4098, Test Loss: 0.3344, Accuracy: 0.8612
Epoch: 49/50, Train Loss: 0.3987, Test Loss: 0.3565, Accuracy: 0.8515
Epoch: 50/50, Train Loss: 0.4069, Test Loss: 0.3285, Accuracy: 0.8635
Finished
Running training. dataset_limit=20000. batch_size=64. lr=0.0005. epochs=50. weight_decay=0.0005
Epoch: 1/50, Train Loss: 0.6768, Test Loss: 0.6144, Accuracy: 0.6740
Epoch: 2/50, Train Loss: 0.6321, Test Loss: 0.5797, Accuracy: 0.7215
Epoch: 3/50, Train Loss: 0.6222, Test Loss: 0.5635, Accuracy: 0.7378
Epoch: 4/50, Train Loss: 0.6153, Test Loss: 0.5583, Accuracy: 0.7378
Epoch: 5/50, Train Loss: 0.6102, Test Loss: 0.5522, Accuracy: 0.7368
Epoch: 6/50, Train Loss: 0.6018, Test Loss: 0.5414, Accuracy: 0.7538
Epoch: 7/50, Train Loss: 0.5990, Test Loss: 0.5516, Accuracy: 0.7530
Epoch: 8/50, Train Loss: 0.5948, Test Loss: 0.5396, Accuracy: 0.7590
Epoch: 9/50, Train Loss: 0.5873, Test Loss: 0.5322, Accuracy: 0.7548
Epoch: 10/50, Train Loss: 0.5825, Test Loss: 0.5084, Accuracy: 0.7752
Epoch: 11/50, Train Loss: 0.5739, Test Loss: 0.5853, Accuracy: 0.6900
Epoch: 12/50, Train Loss: 0.5557, Test Loss: 0.5061, Accuracy: 0.7755
Epoch: 13/50, Train Loss: 0.5508, Test Loss: 0.4860, Accuracy: 0.8055
Epoch: 14/50, Train Loss: 0.5367, Test Loss: 0.4539, Accuracy: 0.8173
Epoch: 15/50, Train Loss: 0.5210, Test Loss: 0.4116, Accuracy: 0.8317
Epoch: 16/50, Train Loss: 0.5095, Test Loss: 0.4111, Accuracy: 0.8470
Epoch: 17/50, Train Loss: 0.5000, Test Loss: 0.4279, Accuracy: 0.8173
Epoch: 18/50, Train Loss: 0.4907, Test Loss: 0.3837, Accuracy: 0.8508
Epoch: 19/50, Train Loss: 0.4800, Test Loss: 0.3803, Accuracy: 0.8588
Epoch: 20/50, Train Loss: 0.4860, Test Loss: 0.4021, Accuracy: 0.8482
Epoch: 21/50, Train Loss: 0.4777, Test Loss: 0.3721, Accuracy: 0.8505
Epoch: 22/50, Train Loss: 0.4718, Test Loss: 0.3640, Accuracy: 0.8555
Epoch: 23/50, Train Loss: 0.4686, Test Loss: 0.3626, Accuracy: 0.8565
Epoch: 24/50, Train Loss: 0.4605, Test Loss: 0.3689, Accuracy: 0.8540
Epoch: 25/50, Train Loss: 0.4638, Test Loss: 0.3684, Accuracy: 0.8518
Epoch: 26/50, Train Loss: 0.4546, Test Loss: 0.3556, Accuracy: 0.8575
Epoch: 27/50, Train Loss: 0.4513, Test Loss: 0.3529, Accuracy: 0.8585
Epoch: 28/50, Train Loss: 0.4474, Test Loss: 0.3599, Accuracy: 0.8568
Epoch: 29/50, Train Loss: 0.4533, Test Loss: 0.3560, Accuracy: 0.8520
Epoch: 30/50, Train Loss: 0.4437, Test Loss: 0.3592, Accuracy: 0.8552
Epoch: 31/50, Train Loss: 0.4382, Test Loss: 0.3494, Accuracy: 0.8570
Epoch: 32/50, Train Loss: 0.4402, Test Loss: 0.3535, Accuracy: 0.8535
Epoch: 33/50, Train Loss: 0.4306, Test Loss: 0.3537, Accuracy: 0.8548
Epoch: 34/50, Train Loss: 0.4277, Test Loss: 0.3751, Accuracy: 0.8407
Epoch: 35/50, Train Loss: 0.4279, Test Loss: 0.3386, Accuracy: 0.8648
Epoch: 36/50, Train Loss: 0.4265, Test Loss: 0.3530, Accuracy: 0.8535
Epoch: 37/50, Train Loss: 0.4276, Test Loss: 0.3515, Accuracy: 0.8510
Epoch: 38/50, Train Loss: 0.4265, Test Loss: 0.3474, Accuracy: 0.8588
Epoch: 39/50, Train Loss: 0.4224, Test Loss: 0.3427, Accuracy: 0.8572
Epoch: 40/50, Train Loss: 0.4177, Test Loss: 0.3465, Accuracy: 0.8612
Epoch: 41/50, Train Loss: 0.4206, Test Loss: 0.3551, Accuracy: 0.8540
Epoch: 42/50, Train Loss: 0.4163, Test Loss: 0.3558, Accuracy: 0.8548
Epoch: 43/50, Train Loss: 0.4118, Test Loss: 0.3998, Accuracy: 0.8267
Epoch: 44/50, Train Loss: 0.4153, Test Loss: 0.3510, Accuracy: 0.8552
Epoch: 45/50, Train Loss: 0.4061, Test Loss: 0.3495, Accuracy: 0.8490
Epoch: 46/50, Train Loss: 0.4102, Test Loss: 0.3769, Accuracy: 0.8375
Epoch: 47/50, Train Loss: 0.4078, Test Loss: 0.3758, Accuracy: 0.8407
Epoch: 48/50, Train Loss: 0.4040, Test Loss: 0.3497, Accuracy: 0.8568
Epoch: 49/50, Train Loss: 0.4031, Test Loss: 0.3763, Accuracy: 0.8423
Epoch: 50/50, Train Loss: 0.4029, Test Loss: 0.3692, Accuracy: 0.8452

Test 10
Running training. dataset_limit=20000. batch_size=64. lr=0.001. epochs=50. weight_decay=0.0005
TODO: 
Epoch: 38/50, Train Loss: 0.4376, Test Loss: 0.3215, Accuracy: 0.8765
Epoch: 48/50, Train Loss: 0.4196, Test Loss: 0.3182, Accuracy: 0.8728

Test 9 (Test 1 with lower weight decay 0.0005):
    # [B, 1, 8000]
    self.cnn_layers = nn.Sequential(  # define CNN layers
        CnnBlock(1, 4, stride=5),  # first cnn block [B, 4, 1600]
        CnnBlock(4, 8, stride=5),  # second cnn block [B, 8, 320]
        CnnBlock(8, 16, stride=5),  # second cnn block [B, 16, 64]
        CnnBlock(16, 32, stride=4),  # second cnn block [B, 32, 16]
    )
    self.fc_layers = nn.Sequential(  # define fully connected layers
        nn.Linear(32 * 16 * 2, 16),  # first linear layer
        nn.ReLU(),  # relu activation
        nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
        nn.Linear(16, 1),  # fourth linera layer
        nn.Sigmoid(),  # sigmoid activation
    )


> Slower train loss convergence. Practically stalled

Running training. dataset_limit=20000. batch_size=64. lr=0.001. epochs=50. weight_decay=0.0005
Epoch [1/50]. Train loss: 0.6443. Test Loss: 0.5962. Precision 0.6679. Recall: 0.8840. F1 score: 0.7609. Accuracy score: 0.7222. 
Epoch [2/50]. Train loss: 0.6141. Test Loss: 0.5663. Precision 0.6872. Recall: 0.8460. F1 score: 0.7584. Accuracy score: 0.7305. 
Epoch [3/50]. Train loss: 0.5982. Test Loss: 0.5524. Precision 0.6863. Recall: 0.8585. F1 score: 0.7628. Accuracy score: 0.7330. 
Epoch [4/50]. Train loss: 0.5871. Test Loss: 0.5576. Precision 0.7011. Recall: 0.8010. F1 score: 0.7477. Accuracy score: 0.7298. 
Epoch [5/50]. Train loss: 0.5833. Test Loss: 0.5539. Precision 0.6558. Recall: 0.9440. F1 score: 0.7739. Accuracy score: 0.7242. 
Epoch [6/50]. Train loss: 0.5691. Test Loss: 0.5210. Precision 0.7064. Recall: 0.9000. F1 score: 0.7916. Accuracy score: 0.7630. 
Epoch [7/50]. Train loss: 0.5637. Test Loss: 0.5146. Precision 0.7068. Recall: 0.8725. F1 score: 0.7809. Accuracy score: 0.7552. 
Epoch [8/50]. Train loss: 0.5567. Test Loss: 0.5051. Precision 0.7119. Recall: 0.8870. F1 score: 0.7898. Accuracy score: 0.7640. 
Epoch [9/50]. Train loss: 0.5452. Test Loss: 0.4986. Precision 0.7143. Recall: 0.8950. F1 score: 0.7945. Accuracy score: 0.7685. 
Epoch [10/50]. Train loss: 0.5378. Test Loss: 0.4879. Precision 0.7254. Recall: 0.9115. F1 score: 0.8079. Accuracy score: 0.7833. 
Epoch [11/50]. Train loss: 0.5307. Test Loss: 0.4724. Precision 0.7582. Recall: 0.8685. F1 score: 0.8096. Accuracy score: 0.7957. 
Epoch [12/50]. Train loss: 0.5255. Test Loss: 0.4848. Precision 0.7407. Recall: 0.9040. F1 score: 0.8142. Accuracy score: 0.7937. 
Epoch [13/50]. Train loss: 0.5208. Test Loss: 0.4535. Precision 0.7465. Recall: 0.9200. F1 score: 0.8242. Accuracy score: 0.8037. 
Epoch [14/50]. Train loss: 0.5062. Test Loss: 0.4556. Precision 0.7483. Recall: 0.9040. F1 score: 0.8188. Accuracy score: 0.8000. 
Epoch [15/50]. Train loss: 0.5017. Test Loss: 0.5057. Precision 0.6933. Recall: 0.9550. F1 score: 0.8034. Accuracy score: 0.7662. 
Epoch [16/50]. Train loss: 0.4904. Test Loss: 0.4403. Precision 0.7716. Recall: 0.8800. F1 score: 0.8222. Accuracy score: 0.8097. 
Epoch [17/50]. Train loss: 0.4874. Test Loss: 0.4934. Precision 0.7066. Recall: 0.9330. F1 score: 0.8041. Accuracy score: 0.7728. 
Epoch [18/50]. Train loss: 0.4760. Test Loss: 0.4168. Precision 0.7842. Recall: 0.9105. F1 score: 0.8427. Accuracy score: 0.8300. 
Epoch [19/50]. Train loss: 0.4586. Test Loss: 0.4443. Precision 0.7744. Recall: 0.9030. F1 score: 0.8338. Accuracy score: 0.8200. 
Epoch [20/50]. Train loss: 0.4652. Test Loss: 0.4305. Precision 0.7795. Recall: 0.9225. F1 score: 0.8450. Accuracy score: 0.8307. 
Epoch [21/50]. Train loss: 0.4536. Test Loss: 0.4161. Precision 0.8076. Recall: 0.8835. F1 score: 0.8438. Accuracy score: 0.8365. 
Epoch [22/50]. Train loss: 0.4522. Test Loss: 0.4073. Precision 0.7814. Recall: 0.9205. F1 score: 0.8453. Accuracy score: 0.8315. 
Epoch [23/50]. Train loss: 0.4397. Test Loss: 0.4097. Precision 0.7867. Recall: 0.9110. F1 score: 0.8443. Accuracy score: 0.8320. 
Epoch [24/50]. Train loss: 0.4396. Test Loss: 0.4433. Precision 0.7519. Recall: 0.9135. F1 score: 0.8248. Accuracy score: 0.8060. 
Epoch [25/50]. Train loss: 0.4374. Test Loss: 0.4040. Precision 0.7997. Recall: 0.9005. F1 score: 0.8471. Accuracy score: 0.8375. 
Epoch [26/50]. Train loss: 0.4313. Test Loss: 0.4734. Precision 0.7031. Recall: 0.9390. F1 score: 0.8041. Accuracy score: 0.7712. 
Epoch [27/50]. Train loss: 0.4279. Test Loss: 0.4066. Precision 0.7907. Recall: 0.8975. F1 score: 0.8407. Accuracy score: 0.8300. 
Epoch [28/50]. Train loss: 0.4227. Test Loss: 0.4581. Precision 0.7570. Recall: 0.9050. F1 score: 0.8244. Accuracy score: 0.8073. 
Epoch [29/50]. Train loss: 0.4187. Test Loss: 0.4388. Precision 0.7571. Recall: 0.9290. F1 score: 0.8343. Accuracy score: 0.8155. 
Epoch [30/50]. Train loss: 0.4135. Test Loss: 0.4200. Precision 0.7551. Recall: 0.9265. F1 score: 0.8321. Accuracy score: 0.8130. 
Epoch [31/50]. Train loss: 0.4074. Test Loss: 0.3973. Precision 0.7874. Recall: 0.8965. F1 score: 0.8384. Accuracy score: 0.8273. 
Epoch [32/50]. Train loss: 0.4034. Test Loss: 0.4254. Precision 0.7738. Recall: 0.9270. F1 score: 0.8435. Accuracy score: 0.8280. 
Epoch [33/50]. Train loss: 0.4021. Test Loss: 0.3885. Precision 0.8033. Recall: 0.9085. F1 score: 0.8527. Accuracy score: 0.8430. 
Epoch [34/50]. Train loss: 0.4038. Test Loss: 0.3901. Precision 0.8093. Recall: 0.8910. F1 score: 0.8482. Accuracy score: 0.8405. 
Epoch [35/50]. Train loss: 0.4000. Test Loss: 0.4334. Precision 0.8014. Recall: 0.8980. F1 score: 0.8470. Accuracy score: 0.8377. 

########################################
########################################
########################################

Test 8 (Test 1 with weight decay 0.001):
    # [B, 1, 8000]
    self.cnn_layers = nn.Sequential(  # define CNN layers
        CnnBlock(1, 4, stride=5),  # first cnn block [B, 4, 1600]
        CnnBlock(4, 8, stride=5),  # second cnn block [B, 8, 320]
        CnnBlock(8, 16, stride=5),  # second cnn block [B, 16, 64]
        CnnBlock(16, 32, stride=4),  # second cnn block [B, 32, 16]
    )
    self.fc_layers = nn.Sequential(  # define fully connected layers
        nn.Linear(32 * 16 * 2, 16),  # first linear layer
        nn.ReLU(),  # relu activation
        nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
        nn.Linear(16, 1),  # fourth linera layer
        nn.Sigmoid(),  # sigmoid activation
    )

Epoch [48/50]. Train loss: 0.3417. Test Loss: 0.3480. Precision 0.8228. Recall: 0.9240. F1 score: 0.8705. Accuracy score: 0.8625. 

> Train loss almost stalled. Test loss was decently following the train loss.
> Let's decrease weight decay one step further 

Epoch [1/50]. Train loss: 0.6509. Test Loss: 0.6345. Precision 0.5800. Recall: 0.9660. F1 score: 0.7248. Accuracy score: 0.6332. 
Epoch [2/50]. Train loss: 0.6090. Test Loss: 0.6519. Precision 0.5523. Recall: 0.9765. F1 score: 0.7056. Accuracy score: 0.5925. 
Epoch [3/50]. Train loss: 0.5908. Test Loss: 0.5415. Precision 0.7256. Recall: 0.8250. F1 score: 0.7721. Accuracy score: 0.7565. 
Epoch [4/50]. Train loss: 0.5584. Test Loss: 0.5259. Precision 0.7214. Recall: 0.8285. F1 score: 0.7712. Accuracy score: 0.7542. 
Epoch [5/50]. Train loss: 0.5382. Test Loss: 0.5594. Precision 0.6452. Recall: 0.9575. F1 score: 0.7709. Accuracy score: 0.7155. 
Epoch [6/50]. Train loss: 0.5186. Test Loss: 0.5318. Precision 0.6652. Recall: 0.9545. F1 score: 0.7840. Accuracy score: 0.7370. 
Epoch [7/50]. Train loss: 0.5091. Test Loss: 0.4782. Precision 0.7185. Recall: 0.9365. F1 score: 0.8131. Accuracy score: 0.7847. 
Epoch [8/50]. Train loss: 0.4952. Test Loss: 0.4583. Precision 0.7373. Recall: 0.9360. F1 score: 0.8249. Accuracy score: 0.8013. 
Epoch [9/50]. Train loss: 0.4869. Test Loss: 0.5300. Precision 0.6583. Recall: 0.9650. F1 score: 0.7826. Accuracy score: 0.7320. 
Epoch [10/50]. Train loss: 0.4737. Test Loss: 0.5460. Precision 0.6709. Recall: 0.9195. F1 score: 0.7758. Accuracy score: 0.7342. 
Epoch [11/50]. Train loss: 0.4653. Test Loss: 0.4130. Precision 0.7731. Recall: 0.9335. F1 score: 0.8458. Accuracy score: 0.8297. 
Epoch [12/50]. Train loss: 0.4554. Test Loss: 0.4250. Precision 0.7593. Recall: 0.9450. F1 score: 0.8421. Accuracy score: 0.8227. 
Epoch [13/50]. Train loss: 0.4474. Test Loss: 0.4030. Precision 0.7850. Recall: 0.9310. F1 score: 0.8518. Accuracy score: 0.8380. 
Epoch [14/50]. Train loss: 0.4436. Test Loss: 0.4027. Precision 0.7754. Recall: 0.9545. F1 score: 0.8557. Accuracy score: 0.8390. 
Epoch [15/50]. Train loss: 0.4451. Test Loss: 0.5168. Precision 0.6736. Recall: 0.9575. F1 score: 0.7908. Accuracy score: 0.7468. 
Epoch [16/50]. Train loss: 0.4407. Test Loss: 0.4125. Precision 0.7713. Recall: 0.9240. F1 score: 0.8408. Accuracy score: 0.8250. 
Epoch [17/50]. Train loss: 0.4319. Test Loss: 0.4316. Precision 0.7413. Recall: 0.9540. F1 score: 0.8343. Accuracy score: 0.8105. 
Epoch [18/50]. Train loss: 0.4325. Test Loss: 0.3574. Precision 0.8194. Recall: 0.9165. F1 score: 0.8652. Accuracy score: 0.8572. 
Epoch [19/50]. Train loss: 0.4268. Test Loss: 0.3756. Precision 0.8149. Recall: 0.9115. F1 score: 0.8605. Accuracy score: 0.8522. 
Epoch [20/50]. Train loss: 0.4158. Test Loss: 0.4607. Precision 0.7227. Recall: 0.9460. F1 score: 0.8194. Accuracy score: 0.7915. 
Epoch [21/50]. Train loss: 0.4098. Test Loss: 0.4274. Precision 0.7403. Recall: 0.9590. F1 score: 0.8355. Accuracy score: 0.8113. 
Epoch [22/50]. Train loss: 0.3985. Test Loss: 0.4081. Precision 0.7546. Recall: 0.9440. F1 score: 0.8387. Accuracy score: 0.8185. 
Epoch [23/50]. Train loss: 0.4001. Test Loss: 0.3641. Precision 0.8021. Recall: 0.9345. F1 score: 0.8633. Accuracy score: 0.8520. 
Epoch [24/50]. Train loss: 0.3917. Test Loss: 0.4802. Precision 0.7087. Recall: 0.9490. F1 score: 0.8115. Accuracy score: 0.7795. 
Epoch [25/50]. Train loss: 0.3897. Test Loss: 0.3956. Precision 0.7732. Recall: 0.9425. F1 score: 0.8495. Accuracy score: 0.8330. 
Epoch [26/50]. Train loss: 0.3909. Test Loss: 0.4337. Precision 0.7333. Recall: 0.9515. F1 score: 0.8283. Accuracy score: 0.8027. 
Epoch [27/50]. Train loss: 0.3849. Test Loss: 0.3884. Precision 0.7853. Recall: 0.9435. F1 score: 0.8571. Accuracy score: 0.8427. 
Epoch [28/50]. Train loss: 0.3832. Test Loss: 0.3517. Precision 0.8250. Recall: 0.9190. F1 score: 0.8694. Accuracy score: 0.8620. 
Epoch [29/50]. Train loss: 0.3843. Test Loss: 0.3469. Precision 0.8346. Recall: 0.9110. F1 score: 0.8711. Accuracy score: 0.8652. 
Epoch [30/50]. Train loss: 0.3805. Test Loss: 0.4227. Precision 0.7335. Recall: 0.9620. F1 score: 0.8324. Accuracy score: 0.8063. 
Epoch [31/50]. Train loss: 0.3699. Test Loss: 0.3874. Precision 0.8115. Recall: 0.8975. F1 score: 0.8523. Accuracy score: 0.8445. 
Epoch [32/50]. Train loss: 0.3733. Test Loss: 0.5030. Precision 0.6722. Recall: 0.9730. F1 score: 0.7951. Accuracy score: 0.7492. 
Epoch [33/50]. Train loss: 0.3732. Test Loss: 0.3565. Precision 0.8377. Recall: 0.8855. F1 score: 0.8610. Accuracy score: 0.8570. 
Epoch [34/50]. Train loss: 0.3674. Test Loss: 0.4575. Precision 0.7118. Recall: 0.9595. F1 score: 0.8173. Accuracy score: 0.7855. 
Epoch [35/50]. Train loss: 0.3657. Test Loss: 0.3604. Precision 0.8018. Recall: 0.9385. F1 score: 0.8648. Accuracy score: 0.8532. 
Epoch [36/50]. Train loss: 0.3668. Test Loss: 0.3346. Precision 0.8320. Recall: 0.9015. F1 score: 0.8654. Accuracy score: 0.8598. 
Epoch [37/50]. Train loss: 0.3595. Test Loss: 0.3516. Precision 0.8105. Recall: 0.9345. F1 score: 0.8681. Accuracy score: 0.8580. 
Epoch [38/50]. Train loss: 0.3590. Test Loss: 0.4057. Precision 0.7546. Recall: 0.9530. F1 score: 0.8422. Accuracy score: 0.8215. 
Epoch [39/50]. Train loss: 0.3547. Test Loss: 0.6120. Precision 0.6177. Recall: 0.9750. F1 score: 0.7563. Accuracy score: 0.6857. 
Epoch [40/50]. Train loss: 0.3545. Test Loss: 0.3874. Precision 0.7872. Recall: 0.9305. F1 score: 0.8529. Accuracy score: 0.8395. 
Epoch [41/50]. Train loss: 0.3544. Test Loss: 0.3365. Precision 0.8241. Recall: 0.9230. F1 score: 0.8708. Accuracy score: 0.8630. 
Epoch [42/50]. Train loss: 0.3429. Test Loss: 0.5556. Precision 0.6407. Recall: 0.9700. F1 score: 0.7717. Accuracy score: 0.7130. 
Epoch [43/50]. Train loss: 0.3477. Test Loss: 0.4895. Precision 0.6893. Recall: 0.9660. F1 score: 0.8045. Accuracy score: 0.7652. 
Epoch [44/50]. Train loss: 0.3364. Test Loss: 0.5286. Precision 0.6733. Recall: 0.9450. F1 score: 0.7864. Accuracy score: 0.7432. 
Epoch [45/50]. Train loss: 0.3495. Test Loss: 0.3879. Precision 0.7906. Recall: 0.9270. F1 score: 0.8534. Accuracy score: 0.8407. 
Epoch [46/50]. Train loss: 0.3365. Test Loss: 0.3401. Precision 0.8283. Recall: 0.9140. F1 score: 0.8690. Accuracy score: 0.8622. 
Epoch [47/50]. Train loss: 0.3442. Test Loss: 0.3611. Precision 0.8143. Recall: 0.9100. F1 score: 0.8595. Accuracy score: 0.8512. 
Epoch [48/50]. Train loss: 0.3417. Test Loss: 0.3480. Precision 0.8228. Recall: 0.9240. F1 score: 0.8705. Accuracy score: 0.8625. 
Epoch [49/50]. Train loss: 0.3372. Test Loss: 0.3961. Precision 0.7917. Recall: 0.9105. F1 score: 0.8470. Accuracy score: 0.8355. 
Epoch [50/50]. Train loss: 0.3407. Test Loss: 0.3758. Precision 0.8215. Recall: 0.8880. F1 score: 0.8534. Accuracy score: 0.8475. 

########################################
########################################
########################################

Test 7 (Test 1 with weight decay 0.01):
    # [B, 1, 8000]
    self.cnn_layers = nn.Sequential(  # define CNN layers
        CnnBlock(1, 4, stride=5),  # first cnn block [B, 4, 1600]
        CnnBlock(4, 8, stride=5),  # second cnn block [B, 8, 320]
        CnnBlock(8, 16, stride=5),  # second cnn block [B, 16, 64]
        CnnBlock(16, 32, stride=4),  # second cnn block [B, 32, 16]
    )
    self.fc_layers = nn.Sequential(  # define fully connected layers
        nn.Linear(32 * 16 * 2, 16),  # first linear layer
        nn.ReLU(),  # relu activation
        nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
        nn.Linear(16, 1),  # fourth linera layer
        nn.Sigmoid(),  # sigmoid activation
    )

> Training stalled. Have to reduce weight decay

Running training. dataset_limit=20000. batch_size=64. lr=0.001. epochs=50. weight_decay=0.01
Epoch [1/50]. Train loss: 0.6596. Test Loss: 0.5989. Precision 0.7212. Recall: 0.6080. F1 score: 0.6598. Accuracy score: 0.6865. 
Epoch [2/50]. Train loss: 0.6186. Test Loss: 0.5607. Precision 0.6985. Recall: 0.7855. F1 score: 0.7395. Accuracy score: 0.7232. 
Epoch [3/50]. Train loss: 0.6061. Test Loss: 0.5637. Precision 0.7057. Recall: 0.8490. F1 score: 0.7708. Accuracy score: 0.7475. 
Epoch [4/50]. Train loss: 0.5870. Test Loss: 0.5658. Precision 0.7022. Recall: 0.8325. F1 score: 0.7618. Accuracy score: 0.7398. 
Epoch [5/50]. Train loss: 0.5805. Test Loss: 0.5445. Precision 0.7194. Recall: 0.8550. F1 score: 0.7814. Accuracy score: 0.7608. 
Epoch [6/50]. Train loss: 0.5764. Test Loss: 0.5300. Precision 0.7082. Recall: 0.8675. F1 score: 0.7798. Accuracy score: 0.7550. 
Epoch [7/50]. Train loss: 0.5672. Test Loss: 0.5311. Precision 0.7137. Recall: 0.8440. F1 score: 0.7734. Accuracy score: 0.7528. 
Epoch [8/50]. Train loss: 0.5688. Test Loss: 0.5488. Precision 0.6894. Recall: 0.8900. F1 score: 0.7770. Accuracy score: 0.7445. 
Epoch [9/50]. Train loss: 0.5629. Test Loss: 0.5282. Precision 0.7017. Recall: 0.9020. F1 score: 0.7893. Accuracy score: 0.7592. 
Epoch [10/50]. Train loss: 0.5629. Test Loss: 0.5196. Precision 0.7115. Recall: 0.8830. F1 score: 0.7880. Accuracy score: 0.7625. 
Epoch [11/50]. Train loss: 0.5616. Test Loss: 0.5383. Precision 0.6857. Recall: 0.9120. F1 score: 0.7828. Accuracy score: 0.7470. 
Epoch [12/50]. Train loss: 0.5568. Test Loss: 0.5253. Precision 0.6948. Recall: 0.8845. F1 score: 0.7783. Accuracy score: 0.7480. 
Epoch [13/50]. Train loss: 0.5558. Test Loss: 0.5278. Precision 0.7037. Recall: 0.9025. F1 score: 0.7908. Accuracy score: 0.7612. 
Epoch [14/50]. Train loss: 0.5525. Test Loss: 0.5324. Precision 0.6925. Recall: 0.9110. F1 score: 0.7869. Accuracy score: 0.7532. 
Epoch [15/50]. Train loss: 0.5415. Test Loss: 0.5062. Precision 0.7163. Recall: 0.8975. F1 score: 0.7967. Accuracy score: 0.7710. 
Epoch [16/50]. Train loss: 0.5371. Test Loss: 0.4882. Precision 0.7540. Recall: 0.8385. F1 score: 0.7940. Accuracy score: 0.7825. 
Epoch [17/50]. Train loss: 0.5344. Test Loss: 0.5013. Precision 0.7231. Recall: 0.9050. F1 score: 0.8039. Accuracy score: 0.7792. 
Epoch [18/50]. Train loss: 0.5328. Test Loss: 0.4873. Precision 0.7442. Recall: 0.8755. F1 score: 0.8045. Accuracy score: 0.7873. 
Epoch [19/50]. Train loss: 0.5290. Test Loss: 0.4886. Precision 0.7325. Recall: 0.8845. F1 score: 0.8014. Accuracy score: 0.7808. 
Epoch [20/50]. Train loss: 0.5327. Test Loss: 0.4941. Precision 0.7215. Recall: 0.8950. F1 score: 0.7989. Accuracy score: 0.7748. 
Epoch [21/50]. Train loss: 0.5246. Test Loss: 0.4759. Precision 0.7478. Recall: 0.8805. F1 score: 0.8087. Accuracy score: 0.7917. 
Epoch [22/50]. Train loss: 0.5229. Test Loss: 0.4720. Precision 0.7357. Recall: 0.9270. F1 score: 0.8204. Accuracy score: 0.7970. 
Epoch [23/50]. Train loss: 0.5204. Test Loss: 0.4761. Precision 0.7567. Recall: 0.8815. F1 score: 0.8143. Accuracy score: 0.7990. 
Epoch [24/50]. Train loss: 0.5160. Test Loss: 0.4722. Precision 0.7350. Recall: 0.9390. F1 score: 0.8246. Accuracy score: 0.8003. 
Epoch [25/50]. Train loss: 0.5143. Test Loss: 0.4652. Precision 0.7670. Recall: 0.8955. F1 score: 0.8263. Accuracy score: 0.8117. 
Epoch [26/50]. Train loss: 0.5124. Test Loss: 0.4548. Precision 0.7691. Recall: 0.9095. F1 score: 0.8334. Accuracy score: 0.8183. 
Epoch [27/50]. Train loss: 0.5105. Test Loss: 0.4751. Precision 0.7530. Recall: 0.9025. F1 score: 0.8210. Accuracy score: 0.8033. 
Epoch [28/50]. Train loss: 0.5061. Test Loss: 0.4470. Precision 0.7898. Recall: 0.8830. F1 score: 0.8338. Accuracy score: 0.8240. 
Epoch [29/50]. Train loss: 0.5044. Test Loss: 0.4445. Precision 0.7669. Recall: 0.9015. F1 score: 0.8288. Accuracy score: 0.8137. 
Epoch [30/50]. Train loss: 0.5008. Test Loss: 0.4537. Precision 0.7493. Recall: 0.9325. F1 score: 0.8309. Accuracy score: 0.8103. 
Epoch [31/50]. Train loss: 0.4962. Test Loss: 0.4452. Precision 0.7435. Recall: 0.9510. F1 score: 0.8346. Accuracy score: 0.8115. 
Epoch [32/50]. Train loss: 0.4966. Test Loss: 0.5172. Precision 0.6859. Recall: 0.9390. F1 score: 0.7927. Accuracy score: 0.7545. 
Epoch [33/50]. Train loss: 0.4938. Test Loss: 0.5177. Precision 0.6794. Recall: 0.9505. F1 score: 0.7924. Accuracy score: 0.7510. 
Epoch [34/50]. Train loss: 0.4956. Test Loss: 0.5084. Precision 0.6864. Recall: 0.9575. F1 score: 0.7996. Accuracy score: 0.7600. 
Epoch [35/50]. Train loss: 0.4990. Test Loss: 0.6119. Precision 0.6038. Recall: 0.9610. F1 score: 0.7417. Accuracy score: 0.6653. 
Epoch [36/50]. Train loss: 0.4924. Test Loss: 0.5907. Precision 0.6140. Recall: 0.9670. F1 score: 0.7511. Accuracy score: 0.6795. 
Epoch [37/50]. Train loss: 0.4932. Test Loss: 0.5283. Precision 0.6637. Recall: 0.9680. F1 score: 0.7875. Accuracy score: 0.7388.

########################################
########################################
########################################

Test 6 (trying 3 CNN layers. with weight decay):
    # [B, 1, 8000]
    self.cnn_layers = nn.Sequential(  # define CNN layers
        CnnBlock(1, 4, stride=4),  # first cnn block [B, 4, 2000]
        CnnBlock(4, 8, stride=4),  # second cnn block [B, 8, 500]
        CnnBlock(8, 16, stride=4),  # second cnn block [B, 16, 125]
    )
    self.fc_layers = nn.Sequential(  # define fully connected layers
        nn.Linear(16 * 125 * 2, 16),  # first linear layer
        nn.ReLU(),  # relu activation
        nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
        nn.Linear(16, 1),  # fourth linera layer
        nn.Sigmoid(),  # sigmoid activation
    )

Epoch [23/50]. Train loss: 0.5099. Test Loss: 0.4579. Precision 0.7573. Recall: 0.9125. F1 score: 0.8277. Accuracy score: 0.8100. 
    
> Slow convergence

Running training. dataset_limit=20000. batch_size=64. lr=0.001
Epoch [1/50]. Train loss: 0.6700. Test Loss: 0.6019. Precision 0.6923. Recall: 0.6435. F1 score: 0.6670. Accuracy score: 0.6787. 
Epoch [2/50]. Train loss: 0.6169. Test Loss: 0.5570. Precision 0.7442. Recall: 0.7100. F1 score: 0.7267. Accuracy score: 0.7330. 
Epoch [3/50]. Train loss: 0.6017. Test Loss: 0.5590. Precision 0.7018. Recall: 0.8580. F1 score: 0.7721. Accuracy score: 0.7468. 
Epoch [4/50]. Train loss: 0.5950. Test Loss: 0.5561. Precision 0.7158. Recall: 0.8210. F1 score: 0.7648. Accuracy score: 0.7475. 
Epoch [5/50]. Train loss: 0.5893. Test Loss: 0.5643. Precision 0.6792. Recall: 0.9220. F1 score: 0.7822. Accuracy score: 0.7432. 
Epoch [6/50]. Train loss: 0.5821. Test Loss: 0.5385. Precision 0.7076. Recall: 0.8725. F1 score: 0.7815. Accuracy score: 0.7560. 
Epoch [7/50]. Train loss: 0.5713. Test Loss: 0.5409. Precision 0.6995. Recall: 0.8870. F1 score: 0.7822. Accuracy score: 0.7530. 
Epoch [8/50]. Train loss: 0.5690. Test Loss: 0.5438. Precision 0.6781. Recall: 0.9270. F1 score: 0.7833. Accuracy score: 0.7435. 
Epoch [9/50]. Train loss: 0.5630. Test Loss: 0.5259. Precision 0.7040. Recall: 0.8885. F1 score: 0.7856. Accuracy score: 0.7575. 
Epoch [10/50]. Train loss: 0.5639. Test Loss: 0.5178. Precision 0.7035. Recall: 0.9240. F1 score: 0.7988. Accuracy score: 0.7672. 
Epoch [11/50]. Train loss: 0.5568. Test Loss: 0.5376. Precision 0.7239. Recall: 0.8310. F1 score: 0.7737. Accuracy score: 0.7570. 
Epoch [12/50]. Train loss: 0.5579. Test Loss: 0.5204. Precision 0.7014. Recall: 0.8890. F1 score: 0.7841. Accuracy score: 0.7552. 
Epoch [13/50]. Train loss: 0.5502. Test Loss: 0.5189. Precision 0.6911. Recall: 0.9130. F1 score: 0.7867. Accuracy score: 0.7525. 
Epoch [14/50]. Train loss: 0.5470. Test Loss: 0.5168. Precision 0.6886. Recall: 0.9220. F1 score: 0.7884. Accuracy score: 0.7525. 
Epoch [15/50]. Train loss: 0.5425. Test Loss: 0.5037. Precision 0.7181. Recall: 0.8710. F1 score: 0.7872. Accuracy score: 0.7645. 
Epoch [16/50]. Train loss: 0.5363. Test Loss: 0.5049. Precision 0.6991. Recall: 0.9190. F1 score: 0.7941. Accuracy score: 0.7618. 
Epoch [17/50]. Train loss: 0.5398. Test Loss: 0.5173. Precision 0.6976. Recall: 0.9240. F1 score: 0.7950. Accuracy score: 0.7618. 
Epoch [18/50]. Train loss: 0.5326. Test Loss: 0.5126. Precision 0.6907. Recall: 0.9335. F1 score: 0.7940. Accuracy score: 0.7578. 
Epoch [19/50]. Train loss: 0.5283. Test Loss: 0.4923. Precision 0.7470. Recall: 0.8860. F1 score: 0.8106. Accuracy score: 0.7930. 
Epoch [20/50]. Train loss: 0.5239. Test Loss: 0.4930. Precision 0.7686. Recall: 0.8320. F1 score: 0.7990. Accuracy score: 0.7907. 
Epoch [21/50]. Train loss: 0.5146. Test Loss: 0.4669. Precision 0.7615. Recall: 0.8810. F1 score: 0.8169. Accuracy score: 0.8025. 
Epoch [22/50]. Train loss: 0.5048. Test Loss: 0.4595. Precision 0.7462. Recall: 0.9010. F1 score: 0.8163. Accuracy score: 0.7973. 
Epoch [23/50]. Train loss: 0.5099. Test Loss: 0.4579. Precision 0.7573. Recall: 0.9125. F1 score: 0.8277. Accuracy score: 0.8100. 
Epoch [24/50]. Train loss: 0.5057. Test Loss: 0.4652. Precision 0.7502. Recall: 0.9160. F1 score: 0.8249. Accuracy score: 0.8055. 
Epoch [25/50]. Train loss: 0.5008. Test Loss: 0.4732. Precision 0.7283. Recall: 0.9420. F1 score: 0.8215. Accuracy score: 0.7953. 

########################################
########################################
########################################


Test 5 (trying 5 cnns with changing strides, increasing lr):
    # [B, 1, 8000]
    self.cnn_layers = nn.Sequential(  # define CNN layers
        CnnBlock(1, 4, stride=4),  # first cnn block [B, 4, 2000]
        CnnBlock(4, 8, stride=5),  # second cnn block [B, 8, 400]
        CnnBlock(8, 16, stride=5),  # second cnn block [B, 16, 80]
        CnnBlock(16, 32, stride=2),  # second cnn block [B, 32, 40]
        CnnBlock(32, 64, stride=2),  # second cnn block [B, 64, 20]
    )
    self.fc_layers = nn.Sequential(  # define fully connected layers
        nn.Linear(64 * 20 * 2, 16),  # first linear layer
        nn.ReLU(),  # relu activation
        nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
        nn.Linear(16, 1),  # fourth linera layer
        nn.Sigmoid(),  # sigmoid activation
    )

Epoch [26/50]. Train loss: 0.4224. Test Loss: 0.3899. Precision 0.8031. Recall: 0.9300. F1 score: 0.8619. Accuracy score: 0.8510. 

> Unstable at some point

Running training. dataset_limit=20000. batch_size=64. lr=0.001
Epoch [1/50]. Train loss: 0.6726. Test Loss: 0.5982. Precision 0.6663. Recall: 0.8515. F1 score: 0.7476. Accuracy score: 0.7125. 
Epoch [2/50]. Train loss: 0.6228. Test Loss: 0.5682. Precision 0.6761. Recall: 0.8955. F1 score: 0.7705. Accuracy score: 0.7332. 
Epoch [3/50]. Train loss: 0.6074. Test Loss: 0.5505. Precision 0.7109. Recall: 0.8460. F1 score: 0.7726. Accuracy score: 0.7510. 
Epoch [4/50]. Train loss: 0.5952. Test Loss: 0.5453. Precision 0.6775. Recall: 0.9210. F1 score: 0.7807. Accuracy score: 0.7412. 
Epoch [5/50]. Train loss: 0.5701. Test Loss: 0.5334. Precision 0.6822. Recall: 0.9230. F1 score: 0.7845. Accuracy score: 0.7465. 
Epoch [6/50]. Train loss: 0.5680. Test Loss: 0.5202. Precision 0.6981. Recall: 0.9075. F1 score: 0.7891. Accuracy score: 0.7575. 
Epoch [7/50]. Train loss: 0.5542. Test Loss: 0.5116. Precision 0.7081. Recall: 0.9025. F1 score: 0.7936. Accuracy score: 0.7652. 
Epoch [8/50]. Train loss: 0.5469. Test Loss: 0.4985. Precision 0.7112. Recall: 0.9110. F1 score: 0.7988. Accuracy score: 0.7705. 
Epoch [9/50]. Train loss: 0.5293. Test Loss: 0.5016. Precision 0.7009. Recall: 0.9315. F1 score: 0.7999. Accuracy score: 0.7670. 
Epoch [10/50]. Train loss: 0.5219. Test Loss: 0.5053. Precision 0.6983. Recall: 0.9165. F1 score: 0.7926. Accuracy score: 0.7602. 
Epoch [11/50]. Train loss: 0.5246. Test Loss: 0.4830. Precision 0.7057. Recall: 0.9495. F1 score: 0.8096. Accuracy score: 0.7768. 
Epoch [12/50]. Train loss: 0.5124. Test Loss: 0.4773. Precision 0.7167. Recall: 0.9310. F1 score: 0.8099. Accuracy score: 0.7815. 
Epoch [13/50]. Train loss: 0.5008. Test Loss: 0.4671. Precision 0.7486. Recall: 0.8995. F1 score: 0.8172. Accuracy score: 0.7987. 
Epoch [14/50]. Train loss: 0.4995. Test Loss: 0.4660. Precision 0.7231. Recall: 0.9415. F1 score: 0.8180. Accuracy score: 0.7905. 
Epoch [15/50]. Train loss: 0.4877. Test Loss: 0.4610. Precision 0.7576. Recall: 0.8875. F1 score: 0.8174. Accuracy score: 0.8017. 
Epoch [16/50]. Train loss: 0.4923. Test Loss: 0.4625. Precision 0.7256. Recall: 0.9295. F1 score: 0.8150. Accuracy score: 0.7890. 
Epoch [17/50]. Train loss: 0.4808. Test Loss: 0.4562. Precision 0.7295. Recall: 0.9265. F1 score: 0.8163. Accuracy score: 0.7915. 
Epoch [18/50]. Train loss: 0.4661. Test Loss: 0.5081. Precision 0.6845. Recall: 0.9545. F1 score: 0.7972. Accuracy score: 0.7572. 
Epoch [19/50]. Train loss: 0.4618. Test Loss: 0.4631. Precision 0.7243. Recall: 0.9325. F1 score: 0.8153. Accuracy score: 0.7887. 
Epoch [20/50]. Train loss: 0.4578. Test Loss: 0.4398. Precision 0.7487. Recall: 0.9340. F1 score: 0.8311. Accuracy score: 0.8103. 
Epoch [21/50]. Train loss: 0.4434. Test Loss: 0.5345. Precision 0.6546. Recall: 0.9655. F1 score: 0.7802. Accuracy score: 0.7280. 
Epoch [22/50]. Train loss: 0.4511. Test Loss: 0.5164. Precision 0.6783. Recall: 0.9310. F1 score: 0.7848. Accuracy score: 0.7448. 
Epoch [23/50]. Train loss: 0.4382. Test Loss: 0.4343. Precision 0.7958. Recall: 0.8630. F1 score: 0.8280. Accuracy score: 0.8207. 
Epoch [24/50]. Train loss: 0.4325. Test Loss: 0.4164. Precision 0.7546. Recall: 0.9535. F1 score: 0.8425. Accuracy score: 0.8217. 
Epoch [25/50]. Train loss: 0.4389. Test Loss: 0.4451. Precision 0.7384. Recall: 0.9345. F1 score: 0.8250. Accuracy score: 0.8017. 
Epoch [26/50]. Train loss: 0.4224. Test Loss: 0.3899. Precision 0.8031. Recall: 0.9300. F1 score: 0.8619. Accuracy score: 0.8510. 
Epoch [27/50]. Train loss: 0.4187. Test Loss: 0.5378. Precision 0.6586. Recall: 0.9580. F1 score: 0.7806. Accuracy score: 0.7308. 
Epoch [28/50]. Train loss: 0.4159. Test Loss: 0.5467. Precision 0.6460. Recall: 0.9680. F1 score: 0.7749. Accuracy score: 0.7188. 
Epoch [29/50]. Train loss: 0.4104. Test Loss: 0.5032. Precision 0.7006. Recall: 0.9430. F1 score: 0.8039. Accuracy score: 0.7700. 
Epoch [30/50]. Train loss: 0.4076. Test Loss: 0.4030. Precision 0.8027. Recall: 0.9075. F1 score: 0.8519. Accuracy score: 0.8423. 
Epoch [31/50]. Train loss: 0.3927. Test Loss: 0.4005. Precision 0.7710. Recall: 0.9460. F1 score: 0.8496. Accuracy score: 0.8325. 
Epoch [32/50]. Train loss: 0.3964. Test Loss: 0.4147. Precision 0.7761. Recall: 0.9360. F1 score: 0.8486. Accuracy score: 0.8330. 
Epoch [33/50]. Train loss: 0.4004. Test Loss: 0.4107. Precision 0.8047. Recall: 0.8980. F1 score: 0.8488. Accuracy score: 0.8400. 
Epoch [34/50]. Train loss: 0.3939. Test Loss: 0.4146. Precision 0.7828. Recall: 0.9280. F1 score: 0.8492. Accuracy score: 0.8353. 
Epoch [35/50]. Train loss: 0.3889. Test Loss: 0.5524. Precision 0.6700. Recall: 0.9430. F1 score: 0.7834. Accuracy score: 0.7392. 
Epoch [36/50]. Train loss: 0.3799. Test Loss: 0.4220. Precision 0.8282. Recall: 0.8845. F1 score: 0.8554. Accuracy score: 0.8505. 
Epoch [37/50]. Train loss: 0.3854. Test Loss: 0.4028. Precision 0.8078. Recall: 0.9165. F1 score: 0.8587. Accuracy score: 0.8492. 
Epoch [38/50]. Train loss: 0.3714. Test Loss: 0.5440. Precision 0.6696. Recall: 0.9435. F1 score: 0.7833. Accuracy score: 0.7390. 


########################################
########################################
########################################


Test 4 (trying 5 cnns):
    # [B, 1, 8000]
    self.cnn_layers = nn.Sequential(  # define CNN layers
        CnnBlock(1, 4, stride=4),  # first cnn block [B, 4, 2000]
        CnnBlock(4, 8, stride=2),  # second cnn block [B, 8, 1000]
        CnnBlock(8, 16, stride=2),  # second cnn block [B, 16, 500]
        CnnBlock(16, 32, stride=5),  # second cnn block [B, 32, 100]
        CnnBlock(32, 64, stride=5),  # second cnn block [B, 64, 20]
    )
    self.fc_layers = nn.Sequential(  # define fully connected layers
        nn.Linear(64 * 20 * 2, 16),  # first linear layer
        nn.ReLU(),  # relu activation
        nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
        nn.Linear(16, 1),  # fourth linera layer
        nn.Sigmoid(),  # sigmoid activation
    )

> Performed worse

Running training. dataset_limit=20000. batch_size=64. lr=0.0005
Epoch [1/50]. Train loss: 0.6589. Test Loss: 0.6109. Precision 0.6813. Recall: 0.7195. F1 score: 0.6999. Accuracy score: 0.6915. 
Epoch [2/50]. Train loss: 0.6185. Test Loss: 0.5654. Precision 0.7028. Recall: 0.8385. F1 score: 0.7647. Accuracy score: 0.7420. 
Epoch [3/50]. Train loss: 0.5982. Test Loss: 0.5423. Precision 0.7014. Recall: 0.9090. F1 score: 0.7918. Accuracy score: 0.7610. 
Epoch [4/50]. Train loss: 0.5890. Test Loss: 0.5467. Precision 0.6921. Recall: 0.8990. F1 score: 0.7821. Accuracy score: 0.7495. 
Epoch [5/50]. Train loss: 0.5813. Test Loss: 0.5219. Precision 0.7238. Recall: 0.8465. F1 score: 0.7804. Accuracy score: 0.7618. 
Epoch [6/50]. Train loss: 0.5753. Test Loss: 0.5206. Precision 0.7196. Recall: 0.8650. F1 score: 0.7856. Accuracy score: 0.7640. 
Epoch [7/50]. Train loss: 0.5614. Test Loss: 0.5260. Precision 0.7224. Recall: 0.8820. F1 score: 0.7942. Accuracy score: 0.7715. 
Epoch [8/50]. Train loss: 0.5554. Test Loss: 0.5009. Precision 0.7394. Recall: 0.8895. F1 score: 0.8075. Accuracy score: 0.7880. 
Epoch [9/50]. Train loss: 0.5393. Test Loss: 0.4987. Precision 0.7132. Recall: 0.9225. F1 score: 0.8044. Accuracy score: 0.7758. 
Epoch [10/50]. Train loss: 0.5399. Test Loss: 0.4884. Precision 0.7372. Recall: 0.9090. F1 score: 0.8142. Accuracy score: 0.7925. 
Epoch [11/50]. Train loss: 0.5300. Test Loss: 0.4677. Precision 0.7529. Recall: 0.9020. F1 score: 0.8207. Accuracy score: 0.8030. 
Epoch [12/50]. Train loss: 0.5236. Test Loss: 0.4754. Precision 0.7371. Recall: 0.9225. F1 score: 0.8195. Accuracy score: 0.7967. 
Epoch [13/50]. Train loss: 0.5140. Test Loss: 0.4632. Precision 0.7571. Recall: 0.8885. F1 score: 0.8176. Accuracy score: 0.8017. 
Epoch [14/50]. Train loss: 0.5068. Test Loss: 0.4705. Precision 0.7443. Recall: 0.9110. F1 score: 0.8192. Accuracy score: 0.7990. 
Epoch [15/50]. Train loss: 0.4965. Test Loss: 0.4514. Precision 0.7532. Recall: 0.9155. F1 score: 0.8265. Accuracy score: 0.8077. 
Epoch [16/50]. Train loss: 0.4994. Test Loss: 0.4547. Precision 0.7694. Recall: 0.9025. F1 score: 0.8306. Accuracy score: 0.8160. 
Epoch [17/50]. Train loss: 0.4885. Test Loss: 0.4516. Precision 0.7683. Recall: 0.8835. F1 score: 0.8219. Accuracy score: 0.8085. 
Epoch [18/50]. Train loss: 0.4835. Test Loss: 0.4532. Precision 0.7923. Recall: 0.8485. F1 score: 0.8194. Accuracy score: 0.8130. 
Epoch [19/50]. Train loss: 0.4751. Test Loss: 0.4471. Precision 0.7527. Recall: 0.9220. F1 score: 0.8288. Accuracy score: 0.8095. 
Epoch [20/50]. Train loss: 0.4687. Test Loss: 0.4474. Precision 0.7582. Recall: 0.9080. F1 score: 0.8264. Accuracy score: 0.8093. 
Epoch [21/50]. Train loss: 0.4601. Test Loss: 0.4515. Precision 0.7530. Recall: 0.9145. F1 score: 0.8259. Accuracy score: 0.8073. 
Epoch [22/50]. Train loss: 0.4599. Test Loss: 0.4587. Precision 0.7782. Recall: 0.8755. F1 score: 0.8240. Accuracy score: 0.8130. 
Epoch [23/50]. Train loss: 0.4565. Test Loss: 0.4442. Precision 0.7713. Recall: 0.8970. F1 score: 0.8294. Accuracy score: 0.8155. 
Epoch [24/50]. Train loss: 0.4462. Test Loss: 0.4649. Precision 0.7388. Recall: 0.9405. F1 score: 0.8275. Accuracy score: 0.8040. 
Epoch [25/50]. Train loss: 0.4414. Test Loss: 0.4791. Precision 0.7224. Recall: 0.9460. F1 score: 0.8192. Accuracy score: 0.7913. 
Epoch [26/50]. Train loss: 0.4399. Test Loss: 0.4521. Precision 0.7607. Recall: 0.9170. F1 score: 0.8316. Accuracy score: 0.8143. 

########################################
########################################
########################################


Test 3 (increased lr):
    # [B, 1, 8000]
    self.cnn_layers = nn.Sequential(  # define CNN layers
        CnnBlock(1, 4, stride=5),  # first cnn block [B, 4, 1600]
        CnnBlock(4, 8, stride=5),  # second cnn block [B, 8, 320]
        CnnBlock(8, 16, stride=5),  # second cnn block [B, 16, 64]
        CnnBlock(16, 32, stride=4),  # second cnn block [B, 32, 16]
    )
    self.fc_layers = nn.Sequential(  # define fully connected layers
        nn.Linear(32 * 16 * 2, 16),  # first linear layer
        nn.ReLU(),  # relu activation
        nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
        nn.Linear(16, 1),  # fourth linera layer
        nn.Sigmoid(),  # sigmoid activation
    )
> Very bumpy, test loss didn't follow train loss closely, sometimes getting but mostly off.


Running training. dataset_limit=20000. batch_size=64. lr=0.005
Epoch [1/25]. Train loss: 0.6478. Test Loss: 0.5727. Precision 0.6620. Recall: 0.8980. F1 score: 0.7621. Accuracy score: 0.7198. 
Epoch [2/25]. Train loss: 0.6104. Test Loss: 0.5756. Precision 0.6404. Recall: 0.9395. F1 score: 0.7617. Accuracy score: 0.7060. 
Epoch [3/25]. Train loss: 0.5931. Test Loss: 0.5591. Precision 0.6698. Recall: 0.8965. F1 score: 0.7667. Accuracy score: 0.7272. 
Epoch [4/25]. Train loss: 0.5925. Test Loss: 0.5514. Precision 0.6749. Recall: 0.9165. F1 score: 0.7774. Accuracy score: 0.7375. 
Epoch [5/25]. Train loss: 0.5810. Test Loss: 0.5527. Precision 0.6650. Recall: 0.9300. F1 score: 0.7755. Accuracy score: 0.7308. 
Epoch [6/25]. Train loss: 0.5689. Test Loss: 0.5595. Precision 0.6502. Recall: 0.9265. F1 score: 0.7641. Accuracy score: 0.7140. 
Epoch [7/25]. Train loss: 0.5602. Test Loss: 0.5439. Precision 0.6592. Recall: 0.9420. F1 score: 0.7756. Accuracy score: 0.7275. 
Epoch [8/25]. Train loss: 0.5377. Test Loss: 0.5078. Precision 0.6845. Recall: 0.9415. F1 score: 0.7927. Accuracy score: 0.7538. 
Epoch [9/25]. Train loss: 0.5270. Test Loss: 0.4769. Precision 0.7266. Recall: 0.9220. F1 score: 0.8127. Accuracy score: 0.7875. 
Epoch [10/25]. Train loss: 0.5144. Test Loss: 0.6136. Precision 0.5987. Recall: 0.9690. F1 score: 0.7401. Accuracy score: 0.6597. 
Epoch [11/25]. Train loss: 0.5078. Test Loss: 0.6735. Precision 0.5641. Recall: 0.9855. F1 score: 0.7175. Accuracy score: 0.6120. 
Epoch [12/25]. Train loss: 0.4942. Test Loss: 0.5467. Precision 0.6477. Recall: 0.9725. F1 score: 0.7775. Accuracy score: 0.7218. 
Epoch [13/25]. Train loss: 0.4938. Test Loss: 0.4735. Precision 0.7070. Recall: 0.9735. F1 score: 0.8191. Accuracy score: 0.7850. 
Epoch [14/25]. Train loss: 0.4851. Test Loss: 0.5816. Precision 0.6173. Recall: 0.9800. F1 score: 0.7575. Accuracy score: 0.6863. 
Epoch [15/25]. Train loss: 0.4736. Test Loss: 0.4489. Precision 0.7320. Recall: 0.9520. F1 score: 0.8276. Accuracy score: 0.8017. 
Epoch [16/25]. Train loss: 0.4744. Test Loss: 0.5872. Precision 0.6099. Recall: 0.9820. F1 score: 0.7525. Accuracy score: 0.6770. 
Epoch [17/25]. Train loss: 0.4729. Test Loss: 0.5724. Precision 0.6251. Recall: 0.9655. F1 score: 0.7589. Accuracy score: 0.6933. 
Epoch [18/25]. Train loss: 0.4635. Test Loss: 0.5534. Precision 0.6407. Recall: 0.9655. F1 score: 0.7702. Accuracy score: 0.7120. 
Epoch [19/25]. Train loss: 0.4617. Test Loss: 0.5211. Precision 0.6760. Recall: 0.9565. F1 score: 0.7921. Accuracy score: 0.7490. 

########################################
########################################
########################################


Test 2 (increased lr):
    # [B, 1, 8000]
    self.cnn_layers = nn.Sequential(  # define CNN layers
        CnnBlock(1, 4, stride=5),  # first cnn block [B, 4, 1600]
        CnnBlock(4, 8, stride=5),  # second cnn block [B, 8, 320]
        CnnBlock(8, 16, stride=5),  # second cnn block [B, 16, 64]
        CnnBlock(16, 32, stride=4),  # second cnn block [B, 32, 16]
    )
    self.fc_layers = nn.Sequential(  # define fully connected layers
        nn.Linear(32 * 16 * 2, 16),  # first linear layer
        nn.ReLU(),  # relu activation
        nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
        nn.Linear(16, 1),  # fourth linera layer
        nn.Sigmoid(),  # sigmoid activation
    )
> More bumpy but reached even better result

Epoch [23/25]. Train loss: 0.3725. Test Loss: 0.3792. Precision 0.7917. Recall: 0.9200. F1 score: 0.8511. Accuracy score: 0.8390. 

Running training. dataset_limit=20000. batch_size=64. lr=0.001
Epoch [1/25]. Train loss: 0.6442. Test Loss: 0.5813. Precision 0.6910. Recall: 0.8320. F1 score: 0.7550. Accuracy score: 0.7300. 
Epoch [2/25]. Train loss: 0.6006. Test Loss: 0.5552. Precision 0.7057. Recall: 0.8130. F1 score: 0.7556. Accuracy score: 0.7370. 
Epoch [3/25]. Train loss: 0.5744. Test Loss: 0.5359. Precision 0.6878. Recall: 0.8890. F1 score: 0.7756. Accuracy score: 0.7428. 
Epoch [4/25]. Train loss: 0.5582. Test Loss: 0.5231. Precision 0.7235. Recall: 0.8045. F1 score: 0.7618. Accuracy score: 0.7485. 
Epoch [5/25]. Train loss: 0.5448. Test Loss: 0.5206. Precision 0.7013. Recall: 0.8805. F1 score: 0.7808. Accuracy score: 0.7528. 
Epoch [6/25]. Train loss: 0.5262. Test Loss: 0.5003. Precision 0.6993. Recall: 0.9185. F1 score: 0.7940. Accuracy score: 0.7618. 
Epoch [7/25]. Train loss: 0.5163. Test Loss: 0.4869. Precision 0.7048. Recall: 0.9215. F1 score: 0.7987. Accuracy score: 0.7678. 
Epoch [8/25]. Train loss: 0.4935. Test Loss: 0.5274. Precision 0.6863. Recall: 0.9125. F1 score: 0.7834. Accuracy score: 0.7478. 
Epoch [9/25]. Train loss: 0.4781. Test Loss: 0.4507. Precision 0.7555. Recall: 0.8960. F1 score: 0.8198. Accuracy score: 0.8030. 
Epoch [10/25]. Train loss: 0.4634. Test Loss: 0.4489. Precision 0.7659. Recall: 0.8915. F1 score: 0.8239. Accuracy score: 0.8095. 
Epoch [11/25]. Train loss: 0.4526. Test Loss: 0.4189. Precision 0.7695. Recall: 0.9245. F1 score: 0.8399. Accuracy score: 0.8237. 
Epoch [12/25]. Train loss: 0.4411. Test Loss: 0.4372. Precision 0.7544. Recall: 0.9200. F1 score: 0.8290. Accuracy score: 0.8103. 
Epoch [13/25]. Train loss: 0.4339. Test Loss: 0.4100. Precision 0.7881. Recall: 0.9095. F1 score: 0.8445. Accuracy score: 0.8325. 
Epoch [14/25]. Train loss: 0.4233. Test Loss: 0.4390. Precision 0.7345. Recall: 0.9390. F1 score: 0.8242. Accuracy score: 0.7997. 
Epoch [15/25]. Train loss: 0.4211. Test Loss: 0.4055. Precision 0.8033. Recall: 0.8780. F1 score: 0.8390. Accuracy score: 0.8315. 
Epoch [16/25]. Train loss: 0.4121. Test Loss: 0.4029. Precision 0.7873. Recall: 0.9125. F1 score: 0.8453. Accuracy score: 0.8330. 
Epoch [17/25]. Train loss: 0.4047. Test Loss: 0.5857. Precision 0.6208. Recall: 0.9715. F1 score: 0.7575. Accuracy score: 0.6890. 
Epoch [18/25]. Train loss: 0.4004. Test Loss: 0.4017. Precision 0.7766. Recall: 0.9210. F1 score: 0.8426. Accuracy score: 0.8280. 
Epoch [19/25]. Train loss: 0.3959. Test Loss: 0.3885. Precision 0.7902. Recall: 0.9060. F1 score: 0.8442. Accuracy score: 0.8327. 
Epoch [20/25]. Train loss: 0.3870. Test Loss: 0.4519. Precision 0.7165. Recall: 0.9615. F1 score: 0.8211. Accuracy score: 0.7905. 
Epoch [21/25]. Train loss: 0.3816. Test Loss: 0.4022. Precision 0.7954. Recall: 0.9000. F1 score: 0.8445. Accuracy score: 0.8343. 
Epoch [22/25]. Train loss: 0.3752. Test Loss: 0.4100. Precision 0.7772. Recall: 0.9085. F1 score: 0.8377. Accuracy score: 0.8240. 
Epoch [23/25]. Train loss: 0.3725. Test Loss: 0.3792. Precision 0.7917. Recall: 0.9200. F1 score: 0.8511. Accuracy score: 0.8390. 
Epoch [24/25]. Train loss: 0.3633. Test Loss: 0.4008. Precision 0.8117. Recall: 0.8770. F1 score: 0.8431. Accuracy score: 0.8367. 
Epoch [25/25]. Train loss: 0.3615. Test Loss: 0.3893. Precision 0.8025. Recall: 0.8960. F1 score: 0.8467. Accuracy score: 0.8377. 
Finished

########################################
########################################
########################################


Test 1 (simplified network with 4 cnns):
    # [B, 1, 8000]
    self.cnn_layers = nn.Sequential(  # define CNN layers
        CnnBlock(1, 4, stride=5),  # first cnn block [B, 4, 1600]
        CnnBlock(4, 8, stride=5),  # second cnn block [B, 8, 320]
        CnnBlock(8, 16, stride=5),  # second cnn block [B, 16, 64]
        CnnBlock(16, 32, stride=4),  # second cnn block [B, 32, 16]
    )
    self.fc_layers = nn.Sequential(  # define fully connected layers
        nn.Linear(32 * 16 * 2, 16),  # first linear layer
        nn.ReLU(),  # relu activation
        nn.Dropout(0.5),  # dropout regularization (50% dropout rate)
        nn.Linear(16, 1),  # fourth linera layer
        nn.Sigmoid(),  # sigmoid activation
    )

Epoch [24/25]. Train loss: 0.4203. Test Loss: 0.4153. Precision 0.7771. Recall: 0.9080. F1 score: 0.8374. Accuracy score: 0.8237. 

> Network regularizes OK, not near to its capacity. 

Running training. dataset_limit=20000. batch_size=64. lr=0.0005
Epoch [1/25]. Train loss: 0.6607. Test Loss: 0.5889. Precision 0.7144. Recall: 0.6765. F1 score: 0.6949. Accuracy score: 0.7030. 
Epoch [2/25]. Train loss: 0.6120. Test Loss: 0.5733. Precision 0.6841. Recall: 0.8305. F1 score: 0.7502. Accuracy score: 0.7235. 
Epoch [3/25]. Train loss: 0.5948. Test Loss: 0.5610. Precision 0.6847. Recall: 0.8545. F1 score: 0.7602. Accuracy score: 0.7305. 
Epoch [4/25]. Train loss: 0.5858. Test Loss: 0.5529. Precision 0.6804. Recall: 0.8865. F1 score: 0.7699. Accuracy score: 0.7350. 
Epoch [5/25]. Train loss: 0.5743. Test Loss: 0.5457. Precision 0.6916. Recall: 0.8410. F1 score: 0.7590. Accuracy score: 0.7330. 
Epoch [6/25]. Train loss: 0.5763. Test Loss: 0.5351. Precision 0.7023. Recall: 0.8565. F1 score: 0.7718. Accuracy score: 0.7468. 
Epoch [7/25]. Train loss: 0.5559. Test Loss: 0.5318. Precision 0.6849. Recall: 0.8910. F1 score: 0.7744. Accuracy score: 0.7405. 
Epoch [8/25]. Train loss: 0.5437. Test Loss: 0.5298. Precision 0.7054. Recall: 0.8430. F1 score: 0.7681. Accuracy score: 0.7455. 
Epoch [9/25]. Train loss: 0.5377. Test Loss: 0.5110. Precision 0.7135. Recall: 0.8715. F1 score: 0.7846. Accuracy score: 0.7608. 
Epoch [10/25]. Train loss: 0.5269. Test Loss: 0.5084. Precision 0.7043. Recall: 0.8860. F1 score: 0.7848. Accuracy score: 0.7570. 
Epoch [11/25]. Train loss: 0.5168. Test Loss: 0.5023. Precision 0.7465. Recall: 0.8335. F1 score: 0.7876. Accuracy score: 0.7752. 
Epoch [12/25]. Train loss: 0.5028. Test Loss: 0.4825. Precision 0.7392. Recall: 0.8915. F1 score: 0.8083. Accuracy score: 0.7885. 
Epoch [13/25]. Train loss: 0.4873. Test Loss: 0.4740. Precision 0.7518. Recall: 0.8845. F1 score: 0.8128. Accuracy score: 0.7963. 
Epoch [14/25]. Train loss: 0.4778. Test Loss: 0.4794. Precision 0.7491. Recall: 0.8985. F1 score: 0.8170. Accuracy score: 0.7987. 
Epoch [15/25]. Train loss: 0.4694. Test Loss: 0.4464. Precision 0.7742. Recall: 0.8830. F1 score: 0.8250. Accuracy score: 0.8127. 
Epoch [16/25]. Train loss: 0.4645. Test Loss: 0.4597. Precision 0.7601. Recall: 0.9060. F1 score: 0.8266. Accuracy score: 0.8100. 
Epoch [17/25]. Train loss: 0.4600. Test Loss: 0.4364. Precision 0.7882. Recall: 0.8800. F1 score: 0.8316. Accuracy score: 0.8217. 
Epoch [18/25]. Train loss: 0.4493. Test Loss: 0.4359. Precision 0.7703. Recall: 0.9120. F1 score: 0.8352. Accuracy score: 0.8200. 
Epoch [19/25]. Train loss: 0.4435. Test Loss: 0.4400. Precision 0.7905. Recall: 0.8885. F1 score: 0.8366. Accuracy score: 0.8265. 
Epoch [20/25]. Train loss: 0.4420. Test Loss: 0.4287. Precision 0.7891. Recall: 0.8810. F1 score: 0.8325. Accuracy score: 0.8227. 
Epoch [21/25]. Train loss: 0.4361. Test Loss: 0.4489. Precision 0.7618. Recall: 0.9100. F1 score: 0.8293. Accuracy score: 0.8127. 
Epoch [22/25]. Train loss: 0.4225. Test Loss: 0.4177. Precision 0.7980. Recall: 0.8650. F1 score: 0.8301. Accuracy score: 0.8230. 
Epoch [23/25]. Train loss: 0.4256. Test Loss: 0.4321. Precision 0.7826. Recall: 0.8730. F1 score: 0.8253. Accuracy score: 0.8153. 
Epoch [24/25]. Train loss: 0.4203. Test Loss: 0.4153. Precision 0.7771. Recall: 0.9080. F1 score: 0.8374. Accuracy score: 0.8237. 
Epoch [25/25]. Train loss: 0.4136. Test Loss: 0.4296. Precision 0.7974. Recall: 0.8795. F1 score: 0.8364. Accuracy score: 0.8280. 


########################################
########################################
########################################


Other improvements:
- Added caching for data (pickle) to make loading faster
"""
