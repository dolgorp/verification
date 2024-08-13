import os #module for os operating, files
import random #used for randomness in the code
import torchaudio #deals with audio data
import torchaudio.transforms as T #for applying audio tranformations
from sklearn.model_selection import train_test_split #splits data

import torch #core pytorch library
import torch.optim as optim #used for updating model weights during training, optimisation algorithms
import torch.nn as nn #classes and tools for neural networks
import torch.nn.functional as F #function for operation of nn like relu, sigmoid
from torch.utils.data import DataLoader, Dataset #load data in batches for traning and creates datasets


def read_as_dict(path, limit): #create new function with two arguments
    result = {} #empty dictionary

    for filename in os.listdir(path): #start loop for all files in the path, lists all files
        if filename == ".DS_Store": #skips system file
            continue
        file_path = os.path.join(path, filename) #creates new var of path and name of the file
        # print(f"Processing file: {file_path}") #used for checking the process

        waveform, original_sample_rate = torchaudio.load(file_path) #loads audio data in waveform and sample rate
        resampler = T.Resample(orig_freq=original_sample_rate, new_freq=8000) #used to convert all sample rates from orig to 8000
        waveform_resampled = resampler(waveform) #standardize all to 8000 

        # Ensure waveform is exactly 8000 samples long
        if waveform_resampled.shape[1] > 8000:
            waveform_resampled = waveform_resampled[:, :8000] #change to 8000 again if not
        elif waveform_resampled.shape[1] < 8000:
            padding_size = 8000 - waveform_resampled.shape[1]
            waveform_resampled = F.pad(waveform_resampled, (0, padding_size)) #if its shorter it fills with 0 to reach 8000

        username = filename.split("_")[0] #extract userid from file name
        if username in result: 
            result[username].append(waveform_resampled) #if user exist in dictionary, add file to it
        else:
            result[username] = [waveform_resampled] #add user and file into dict
        if len(result) >= limit: #stop if dict hits the limit per user
            break

    return result #return dict


def to_positive_pairs(user_to_audio, limit): #new function that takes dict and limit number of rows for matching samples
    # construct queue of user samples
    pairs = [] #empty list to store all pairs
    for user, audios in user_to_audio.items(): #loop through each user's audio samples
        user_pairs = [] #list to hold pairs for current user in the loop
        for i in range(len(audios)): #loop through each audio sample
            for j in range(i + 1, len(audios)): #loop thorugh next audio samples
                if len(user_pairs) < 5: #set limit to 5 pairs per users
                    user_pairs.append((audios[i], audios[j], 1)) #add to list and put label 1
                else: #?if list is full
                    break #?break the cycle
            if len(user_pairs) >= limit:  #?
                break
        pairs.extend(user_pairs) #add user pairs to the main list

    random.Random(42).shuffle(pairs) #randomly shuffle all paris

    return pairs[:limit] #?return limited number of pairs


def to_negative_pairs(user_to_audio, limit): #set fnction to create negative pairs
    neg_pairs = [] #empty list to sore

    # construct queue of user samples
    index_user_sample = [] #to hold user pairs
    for user, audios in user_to_audio.items(): #loop through users
        for audio in audios: #loop through audios
            index_user_sample.append((user, audio)) #store pairs in the list

    queue1 = list(index_user_sample) #copy user audio list
    random.Random(42).shuffle(queue1) #randomly shuffle new list

    queue2 = list(index_user_sample) #another copy of user audio list
    random.Random(24).shuffle(queue2) #randomly shuffle new list
 
    for i in range(len(queue1)): #loop through queue lists
        if queue1[i][0] != queue2[i][0]: #check if users are same
            neg_pairs.append((queue1[i][1], queue2[i][1], 0)) #add a pair if they are not same and set label 0

            if len(neg_pairs) == limit: #break if list hit limit
                break

    return neg_pairs #return list with negative pairs


class VoiceDataset(Dataset): #dataset class for audio data
    def __init__(self, pairs): #initiate, intake triples
        self.pairs = pairs #store triples

    def __len__(self): #length method
        return len(self.pairs) #returns number of triples

    def __getitem__(self, idx): #new method
        spectrogram1, spectrogram2, label = self.pairs[idx] #assign variables
        # [1, 8000], [1, 8000], [1]
        return spectrogram1, spectrogram2, torch.tensor(label, dtype=torch.float32) #returns the pair of spectrograms and label as a tensor


class CnnBlock(nn.Module): #make CNN block class
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1): #initialize with hyperparametrs
        super(CnnBlock, self).__init__() #call the parent class constructor - module
        self.conv1 = nn.Conv1d( #first convolutional layer with parameters
            in_channels, # ToDo: read about convolution more
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels) #batch normalization after first convolution
        self.conv2 = nn.Conv1d( #second convolutional layer
            out_channels, out_channels, kernel_size=kernel, stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels) #batch normalization after second convolution

        # Residual connection
        # ToDo: explain why it is needed for CNN (hint: signal decay)
        self.shortcut = nn.Sequential() #make an empty sequential layer for the shortcut connection
        if stride != 1 or in_channels != out_channels: #if stride or channel count changes, adjust the shortcut with a 1x1 convolution and batch norm
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        self.relu = nn.ReLU() #relu activated

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) #apply the first conv layer, batch norm, and ReLU activation
        out = self.bn2(self.conv2(out)) #apply the second conv layer and batch norm
        out = out + self.shortcut(x) #add the shortcut to the output
        out = F.relu(out) #activate relu to output
        return out


class VoiceMatchModel(nn.Module): #new class
    def __init__(self):
        super(VoiceMatchModel, self).__init__() #call the parent class constructor - voicematchmodule
        # [B, 1, 8000]
        self.cnn_layers = nn.Sequential( #define CNN layers
            CnnBlock(1, 16),  # first cnn block [B, 16, 8000]
            CnnBlock(16, 32, stride=2),  # second cnn block [B, 32, 4000]
            CnnBlock(32, 64, stride=2),  # third cnn block [B, 64, 2000]
            CnnBlock(64, 128, stride=2),  # fourth cnn block [B, 128, 1000]
        )

        # [B, 128 * 1000 * 2] -  incoming shape
        self.fc_layers = nn.Sequential( #define fully connected layers
            nn.Linear(128 * 1000 * 2, 256), #first linear layer
            nn.ReLU(), #relu activation
            nn.Dropout(0.5), #dropout regularization (50% dropout rate)
            nn.Linear(256, 128), #second linera layer
            nn.ReLU(), #relu activation
            nn.Dropout(0.5), #dropout regularization (50% dropout rate)
            nn.Linear(128, 64), #third linera layer
            nn.ReLU(), #relu activation
            nn.Dropout(0.5), #dropout regularization (50% dropout rate)
            nn.Linear(64, 1), #fourth linera layer
            nn.Sigmoid(), #sigmoid activation
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: [B, 1, 8000]
        # x2: [B, 1, 8000]

        # Forward each through CNN layers
        x1 = self.cnn_layers(x1) #pass the first input through the CNN layers
        x2 = self.cnn_layers(x2) #pass the second input through the CNN layers\

        # x1: [B, 128, 1000]
        # x2: [B, 128, 1000]

        # Concatenate together
        x1 = x1.view(x1.size(0), -1) #flatten x1 [B, 128 * 1000]
        x2 = x2.view(x2.size(0), -1) #flatten x2 [B, 128 * 1000]
        x = torch.cat((x1, x2), dim=1) #concatenate along the feature dimension: [B, 128 * 1000 * 2]


        # Forward through linear (fully-connected) layers
        x = self.fc_layers(x) #pass through the fully connected layers to get the final output
        # x: [B, 1]
        return x #return the final output (probability of a match)


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10): #create function for training
    model.train() #set the model to training mode
    with torch.autograd.set_detect_anomaly(True): #enable anomaly detection for debugging
        for epoch in range(num_epochs): #loop over the number of epochs
            running_loss = 0.0 #set running loss for the epoch
            for audio1, audio2, labels in train_loader: #iterate over batches of data in the training set
                # audio1: [B, 1, 8000]
                # audio2: [B, 1, 8000]
                # labels: [B]

                optimizer.zero_grad() #zero the gradients before each forward pass

                # Move data to the appropriate device
                audio1 = audio1.to(device)
                audio2 = audio2.to(device)
                labels = labels.to(device)

                # Forward pass through the model
                # audio1: [B, 1, 8000]
                # audio2: [B, 1, 8000]
                # labels: [B]
                outputs = model(audio1, audio2)
                # outputs: [B, 1]
                outputs = outputs.squeeze(1) #remove the single-dimensional entry from the output
                # outputs: [B]

                loss = criterion(outputs, labels) #calculate the loss
                # [B]

                # Backward pass and optimize
                loss.backward() #compute gradients for backpropagation
                optimizer.step() #update the model parameters

                running_loss += loss.item() * audio1.size(0) #calculate running loss

            epoch_loss = running_loss / len(train_loader.dataset) #calculate average loss for epoch

            # Evaluate on test data
            model.eval()  # Set model to evaluation mode
            test_loss = 0.0 #initialize test loss
            all_outputs = [] #list to store all predictions
            all_labels = [] #list to store all true labels
            with torch.no_grad(): #disable gradient computation for testing
                for audio1, audio2, labels in test_loader: #iterate over the test data
                    # audio1: [B, 1, 8000]
                    # audio2: [B, 1, 8000]
                    # labels: [B]

                    audio1 = audio1.to(device)
                    audio2 = audio2.to(device)
                    labels = labels.to(device)

                    outputs = model(audio1, audio2).squeeze(1) #forward pass and remove single dimension
                    # outputs: [B]
                    loss = criterion(outputs, labels)  #calculate the test loss
                    # [B]

                    test_loss += loss.item() * audio1.size(0) #calculate test loss

                    all_outputs.append(outputs) #store predictions
                    all_labels.append(labels) #store true labels

            test_loss /= len(test_loader.dataset) #calculate average test loss
            all_outputs = torch.cat(all_outputs) #put all predictions into a single tensor
            all_labels = torch.cat(all_labels) #put all true labels into a single tensor
            precision, recall, f1_score, accuracy = evaluate(all_outputs, all_labels) #evaluate precision, recall, and F1 score
            print( #print the metrics for the current epoch
                f"Epoch [{epoch+1}/{num_epochs}]. Train loss: {epoch_loss:.4f}. Test Loss: {test_loss:.4f}. "
                f"Precision {precision:.4f}. Recall: {recall:.4f}. F1 score: {f1_score:.4f}. Accuracy score: {accuracy:.4f}. "
            )
            model.train()  # Set model back to train mode


def evaluate(predictions, labels):
    true_positives = torch.logical_and(predictions >= 0.5, labels == 1).sum().item() #calculate true positives
    false_positives = torch.logical_and(predictions >= 0.5, labels == 0).sum().item() #calculate false positives
    true_negatives = torch.logical_and(predictions < 0.5, labels == 0).sum().item() # calculate true negatives
    false_negatives = torch.logical_and(predictions < 0.5, labels == 1).sum().item() #calculate false negatives

    #calculate precision: the ratio of true positives to the total predicted positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )

    #calculate recall: the ratio of true positives to the total actual positives
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0
    )

    #calculate F1 score: the harmonic mean of precision and recall
    f1_score = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )

    # calculate accuracy: the ratio of correct predictions (true positives + true negatives) to the total predictions
    accuracy = (
        (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
        if true_positives + false_positives + true_negatives + false_negatives > 0
        else 0
    )

    # return results
    return precision, recall, f1_score, accuracy


# Prepare data
print("Preparing data")
limit = 1000 #set a limit on the total number of pairs to be generated
# one = read_as_dict("/root/learning-nn/resources/speech_commands/one", 1000)

#load data into a dict from marvin folder
one = read_as_dict("data_raw/seven", 1000)

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