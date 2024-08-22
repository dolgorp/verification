import os
import random
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def read_as_dict(path, limit):
    result = {}

    for filename in os.listdir(path):
        if filename == ".DS_Store":
            continue
        file_path = os.path.join(path, filename)
        # print(f"Processing file: {file_path}")

        waveform, original_sample_rate = torchaudio.load(file_path)
        resampler = T.Resample(orig_freq=original_sample_rate, new_freq=8000)
        waveform_resampled = resampler(waveform)

        # Ensure waveform is exactly 8000 samples long
        if waveform_resampled.shape[1] > 8000:
            waveform_resampled = waveform_resampled[:, :8000]
        elif waveform_resampled.shape[1] < 8000:
            padding_size = 8000 - waveform_resampled.shape[1]
            waveform_resampled = F.pad(waveform_resampled, (0, padding_size))

        username = filename.split("_")[0]
        if username in result:
            result[username].append(waveform_resampled)
        else:
            result[username] = [waveform_resampled]
        if len(result) >= limit:
            break

    return result


def to_positive_pairs(user_to_audio, limit):
    # construct queue of user samples
    pairs = []
    for user, audios in user_to_audio.items():
        user_pairs = []
        for i in range(len(audios)):
            for j in range(i + 1, len(audios)):
                if len(user_pairs) < 5:
                    user_pairs.append((audios[i], audios[j], 1))
                else:
                    break
            if len(user_pairs) >= limit:  # repeat break for each loop
                break
        pairs.extend(user_pairs)

    random.Random(42).shuffle(pairs)

    return pairs[:limit]


def to_negative_pairs(user_to_audio, limit):
    neg_pairs = []

    # construct queue of user samples
    index_user_sample = []
    for user, audios in user_to_audio.items():
        for audio in audios:
            index_user_sample.append((user, audio))

    queue1 = list(index_user_sample)
    random.Random(42).shuffle(queue1)

    queue2 = list(index_user_sample)
    random.Random(24).shuffle(queue2)

    for i in range(len(queue1)):
        if queue1[i][0] != queue2[i][0]:
            neg_pairs.append((queue1[i][1], queue2[i][1], 0))

            if len(neg_pairs) == limit:
                break

    return neg_pairs


class VoiceDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        spectrogram1, spectrogram2, label = self.pairs[idx]
        return spectrogram1, spectrogram2, torch.tensor(label, dtype=torch.float32)


class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
        super(CnnBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Residual connection
        # ToDo: explain why it is needed for CNN (hint: signal decay)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class VoiceMatchModel(nn.Module):
    def __init__(self):
        super(VoiceMatchModel, self).__init__()
        # [B, 1, 8000]
        self.cnn_layers = nn.Sequential(
            CnnBlock(1, 16),  # [B, 16, 8000]
            CnnBlock(16, 32, stride=2),  # [B, 32, 4000]
            CnnBlock(32, 64, stride=2),  # [B, 64, 2000]
            CnnBlock(64, 128, stride=2),  # [B, 128, 1000]
        )

        # [B, 128 * 1000 * 2]
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 1000 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # Forward each through CNN layers
        x1 = self.cnn_layers(x1)
        x2 = self.cnn_layers(x2)

        # Concatenate together
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)

        # Forward through linear (fully-connected) layers
        x = self.fc_layers(x)
        return x


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    model.train()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for audio1, audio2, labels in train_loader:
                optimizer.zero_grad()

                # Move data to the appropriate device
                audio1 = audio1.to(device)
                audio2 = audio2.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(audio1, audio2)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * audio1.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            # Evaluate on test data
            model.eval()  # Set model to evaluation mode
            test_loss = 0.0
            all_outputs = []
            all_labels = []
            with torch.no_grad():
                for audio1, audio2, labels in test_loader:
                    audio1 = audio1.to(device)
                    audio2 = audio2.to(device)
                    labels = labels.to(device)

                    outputs = model(audio1, audio2).squeeze(1)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * audio1.size(0)

                    all_outputs.append(outputs)
                    all_labels.append(labels)

            test_loss /= len(test_loader.dataset)
            all_outputs = torch.cat(all_outputs)
            all_labels = torch.cat(all_labels)
            precision, recall, f1_score = evaluate(all_outputs, all_labels)
            print(
                f"Epoch [{epoch+1}/{num_epochs}]. Train loss: {epoch_loss:.4f}. Test Loss: {test_loss:.4f}. "
                f"Precision {precision:.4f}. Recall: {recall:.4f}. F1 score: {f1_score:.4f}. "
            )
            model.train()  # Set model back to train mode


def evaluate(predictions, labels):
    true_positives = torch.logical_and(predictions >= 0.5, labels == 1).sum().item()
    false_positives = torch.logical_and(predictions >= 0.5, labels == 0).sum().item()
    false_negatives = torch.logical_and(predictions < 0.5, labels == 1).sum().item()

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0
    )
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )
    return precision, recall, f1_score


# Prepare data
print("Preparing data")
limit = 768
# one = read_as_dict("/root/learning-nn/resources/speech_commands/one", 1000)
one = read_as_dict("data_raw/marvin", 1000)

positive_pairs = to_positive_pairs(one, limit=int(limit * 0.5))
print(f"Read {len(positive_pairs)} positive_pairs")

negative_pairs = to_negative_pairs(one, limit=int(limit * 0.5))
print(f"Read {len(negative_pairs)} negative_pairs")

p_train, p_test = train_test_split(positive_pairs, test_size=0.2, random_state=42)
n_train, n_test = train_test_split(negative_pairs, test_size=0.2, random_state=42)

train = p_train + n_train
test = p_test + n_test

random.Random(42).shuffle(train)
random.Random(42).shuffle(test)

# Setting up the training
print("Running training")
device = torch.device("cpu")
model = VoiceMatchModel().to(device)
train_loader = DataLoader(dataset=VoiceDataset(train), batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=VoiceDataset(test), batch_size=64, shuffle=True)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50)

print("Finished")