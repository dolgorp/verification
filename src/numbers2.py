from datetime import datetime
import os  # module for os operating, files
import random  # used for randomness in the code
import torchaudio  # deals with audio data
import torchaudio.transforms as T  # for applying audio tranformations
from sklearn.model_selection import train_test_split  # splits data

import gradio as gr
import torch  # core pytorch library
import torch.optim as optim  # used for updating model weights during training, optimisation algorithms
import torch.nn as nn  # classes and tools for neural networks
import torch.nn.functional as F  # function for operation of nn like relu, sigmoid
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # load data in batches for traning and creates datasets
from voice_main import VoiceModel
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

class NumbersDataset(Dataset):

    def __init__(self, audio, label):
        assert audio.size(0) == label.size(0)
        self.audio = audio
        self.label = label

    def __getitem__(self, index): 
        return (self.audio[index], self.label[index]) #tuples
    
    def __len__(self):
        return self.audio.size(dim=0) #assuming audio and label tensors are same size


class NumbersModel(nn.Module):

    def __init__(self, checkpoint_path = None):
        super().__init__()
        # [B, 1, 8000]
        self.cnn1 = BlockCNN(in_channels=1, out_channels=16, stride = 1, kernel_size= 3, padding = 1)
        # [B, 16, 8000]
        self.cnn2 = BlockCNN(in_channels=16, out_channels=32, stride = 2, kernel_size = 3, padding = 1)
        # [B, 32, 4000]
        self.cnn3 = BlockCNN(in_channels=32, out_channels=64, stride = 2, kernel_size = 3, padding = 1)
        # [B, 64, 2000]
        self.cnn4 = BlockCNN(in_channels=64, out_channels=128, stride = 2, kernel_size = 3, padding = 1)
        # [B, 128, 1000]
        self.cnn5 = BlockCNN(in_channels=128, out_channels=256, stride = 2, kernel_size = 3, padding = 1)
        # [B, 256, 500]

        #self.flatten = nn.Flatten()

        self.fc1 =  nn.Linear(in_features=256 * 500, out_features=256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout1d(p=0.5)

        self.fc2 =  nn.Linear(in_features=256, out_features=128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout1d(p=0.5)

        self.fc3 =  nn.Linear(in_features=128, out_features=64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout1d(p=0.5)

        self.fc4 =  nn.Linear(in_features=64, out_features=10)

        #self.softmax = nn.Softmax(dim=1)

        if checkpoint_path:
            self.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    
    def forward(self, audio):
        x = self.cnn1(audio)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)

        x = torch.reshape(x, (x.size(0), x.size(1) * x.size(2)))

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)

        return x


        """
        Architecture
        audio > 5 cnn blocks > 4 FC layers > [10]

        1. audio [B, 1, 8000] > 
        2. CNN1: audio [B, 16, 8000] (params in: 1, out: 16 ; stride = 1, kernel = 3, padding = 1)
        3. CNN2: audio [B, 32, 4000] > (params in: 16, out: 32 ; stride = 2, kernel = 3, padding = 1)
        4. CNN3: audio [B. 64, 2000] > (params in: 32, out: 64 ; stride = 2, kernel = 3, padding = 1)
        5. CNN4: audio [B. 128, 1000] > (params in: 64, out: 128 ; stride = 2, kernel = 3, padding = 1)
        6. CNN5: audio [B, 256, 500] (params in: 128, out: 256 ; stride = 2, kernel = 3, padding = 1)
        7. Flattening to [B, 256 * 500] 
        8. FC1: audio [B, 256] (params in: 256 * 500, out: 256)
        AF: ReLu, Drop out = nn.Dropout(p=0.5)
        9. FC2: audio [B, 128] (params in: 256, out: 128)
        AF: ReLu, Drop out = nn.Dropout(p=0.5)
        10. FC3: audio [B, 64] (params in: 128, out: 64)
        AF: ReLu, Drop out = nn.Dropout(p=0.5)
        11. FC4: audio [B, 10] (params in: 64, out: 10)
        """
class BlockCNN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding): 
        super().__init__()
        self.cnn1 = nn.Conv1d(in_channels ,out_channels, kernel_size, stride, padding)
        self.batchnorm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.cnn2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        self.shortcut_cnn = nn.Conv1d(in_channels ,out_channels, kernel_size, stride, padding)
        self.shortcut_batchnorm = nn.BatchNorm1d(out_channels)


    def forward(self, audio):
        x = self.cnn1(audio)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.cnn2(x)
        x = self.batchnorm2(x)
        x = x + self.shortcut_batchnorm(self.shortcut_cnn(audio))
        x = self.relu2(x)
        return x

    """
    CNN Block1:
    [B, 1, 8000]
    1. CNN [B, 16, 8000] (params in: 1, out: 16 ; stride = 1, kernel = 3, padding = 1)
    2. BatchNorm layer [B, 16, 8000] - number of features (params channels: 16)
        ReLu
    3. CNN [B, 16, 8000] (params in: 16, out: 16 ; stride = 1, kernel = 3, padding = 1)
    4. BatchNorm layer [B, 16, 8000] - number of features (params channels: 16)
        ReLu
    5. Residual Connection - [B, 16, 8000] + shortcut([B, 1, 8000])
        - shortcut (CNN 1-16, stride = 1) + batchnorm

    CNN Block2:
    [B, 16, 8000]
    1. CNN [B, 32, 4000] (params in: 16, out: 32 ; stride = 2, kernel = 3, padding = 1)
    2. BatchNorm layer [B, 32, 4000] - number of features (params channels: 32)
        ReLu
    3. CNN [B, 32, 4000] (params in: 32, out: 32 ; stride = 1, kernel = 3, padding = 1)
    4. BatchNorm layer [B, 32, 4000] - number of features (params channels: 32)
        ReLu
    5. Residual Connection - [B, 32, 4000] + shortcut([B, 16, 8000])
        - shortcut (CNN 16-32, stride = 2) + batchnorm
    """

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, checkpoint_path=None):
    timestamp = datetime.now().isoformat()

    model.train() #set model into training mode
    for epoch in range(num_epochs):
        running_loss = 0
        for audio, label in train_loader:
            # audio shape is [B, 8000]
            audio = audio.to(DEVICE)
            label = label.to(DEVICE)
          
            optimizer.zero_grad() #reset optimizer
            prediction = model(audio)
            loss = criterion(prediction, label)
            loss.backward() #calculating updates (derivatives) for weights and biases based on loss
            optimizer.step() #use updates and apply to the model
            running_loss += loss * audio.size(dim=0)

        # calculate average loss for epoch
        train_loss = running_loss / len(train_loader.dataset)


        model.eval()

        predictions = []
        labels = []

        running_loss = 0
        with torch.no_grad():
            for audio, label in test_loader:
                # audio shape is [B, 8000]
                audio = audio.to(DEVICE)
                label = label.to(DEVICE)
            
                prediction = model(audio) # [B, 10]
                loss = criterion(prediction, label) 
    
                running_loss += loss * audio.size(dim=0)

                prediction_distribution = F.softmax(prediction, dim=1)
                predictions.append(prediction_distribution)
                labels.append(label)
            

        # calculate average loss for epoch
        test_loss = running_loss / len(test_loader.dataset)

        predictions = torch.cat(predictions)  # put all predictions into a single tensor
        labels = torch.cat(labels)  # put all true labels into a single tensor
        accuracy, f1, conf_matrix = evaluate(predictions, labels)

        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

        if checkpoint_path:
            name = os.path.join(checkpoint_path, f'{timestamp}_e{epoch+1}_tr{train_loss:.4f}_te{test_loss:.4f}_ac{accuracy:.4f}.pth')
            torch.save(model.state_dict(), name)
        model.train()

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


def run_test(dataset_limit: int, batch_size: int, lr: int, model_checkpoint_path=None, train_checkpoint_path=None):
    # Prepare data
    print("Preparing data")
    # set a limit on the total number of pairs to be generated
    # one = read_as_dict("/root/learning-nn/resources/speech_commands/one", 1000)

    # load data into a dict from marvin folder
    audio_samples_tensor, labels_tensor = to_pairs(
        "/Users/Dolg/Documents/Flatiron/capstone/data_numbers_wav", dataset_limit
    )
    print(f"Read {len(audio_samples_tensor)} pairs")

    as_train, as_test, l_train, l_test = train_test_split(
        audio_samples_tensor, labels_tensor, test_size=0.2, random_state=42
    )

    # Setting up the training
    print(
        f"Running training. dataset_limit={dataset_limit}. batch_size={batch_size}. lr={lr}"
    )
    model = NumbersModel(model_checkpoint_path).to(DEVICE)
    train_loader = DataLoader(
        dataset=NumbersDataset(as_train, l_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=NumbersDataset(as_test, l_test), batch_size=batch_size, shuffle=True
    )

    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss() #log loss, criterion for assesing classification
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, checkpoint_path=train_checkpoint_path)

    print("Finished")


"""
dataset_limit = 10_000  batch_size = 64    lr = 0.0005          Epoch [48/50]. Train loss: 0.5297. Test Loss: 1.3183. Accuracy 0.6765. F1: 0.6769.

dataset_limit = 20_000  batch_size = 16    lr = 0.0005         *Epoch [16/50]. Train loss: 0.6332. Test Loss: 0.9068. Accuracy 0.7007. F1: 0.7027.
dataset_limit = 20_000  batch_size = 32    lr = 0.0005          Epoch [16/50]. Train loss: 0.4637. Test Loss: 0.7815. Accuracy 0.7927. F1: 0.7931.
dataset_limit = 20_000  batch_size = 64    lr = 0.0005          Epoch [26/50]. Train loss: 0.5147. Test Loss: 0.8011. Accuracy 0.7688. F1: 0.7694.

dataset_limit = 20_000  batch_size = 256   lr = 0.0005          Epoch [35/50]. Train loss: 0.7958. Test Loss: 1.1784. Accuracy 0.6048. F1: 0.5849. 
"""



# for dataset_limit in [20000]:
#     for batch_size in [32]:
#         for lr in [0.0005]:
#             run_test(
#                 dataset_limit,
#                 batch_size,
#                 lr,
#                 model_checkpoint_path="/Users/Dolg/Documents/Flatiron/capstone/model_checkpoints/2024_08_20T05_22_57_318460_e_30_tr_2_0492_te_1_4213_ac_0_8315.pth",
#                 train_checkpoint_path="/Users/Dolg/Documents/Flatiron/capstone/checkpoints",
#             )


def to_waveform(audio):
    sample_rate, waveform = audio
    waveform = torch.tensor(waveform, dtype=torch.float32)
    resampler = T.Resample(
        orig_freq=sample_rate, new_freq=8000
    )  # used to convert all sample rates from orig to 8000
    waveform_resampled = resampler(waveform)  # standardize all to 8000

    # Ensure waveform is exactly 8000 samples long
    if waveform_resampled.shape[0] > 8000:
        # change to 8000 again if not
        waveform_resampled = waveform_resampled[:8000] 
    elif waveform_resampled.shape[0] < 8000:
        padding_size = 8000 - waveform_resampled.shape[0]
        waveform_resampled = F.pad(
            waveform_resampled, (0, padding_size)
        )  # if its shorter it fills with 0 to reach 8000
    return waveform_resampled.unsqueeze(0).unsqueeze(0)

def server():
    # 2024_08_20T05_22_57_318460_e_30_tr_2_0492_te_1_4213_ac_0_8315.pth
    numbers_checkpoint_path = "/Users/Dolg/Documents/Flatiron/capstone/model_checkpoints/numbers_model_checkpoints.pth"
    numbers_model = NumbersModel(numbers_checkpoint_path).to(DEVICE)
    numbers_model.eval()

    # 2024-08-20T15_40_02.942877_e49_tr0.4334_te0.3812_ac0.8420.pth
    voice_checkpoint_path = "/Users/Dolg/Documents/Flatiron/capstone/model_checkpoints/voice_model_checkpoints.pth"
    voice_model = VoiceModel(voice_checkpoint_path).to(DEVICE)
    voice_model.eval()

    def recognize_number(audio):
        nonlocal numbers_model
        if audio:
            waveform = to_waveform(audio)
            prediction = numbers_model(waveform)
            result = torch.argmax(F.softmax(prediction, dim=1)).item()
        else:
            result = '-'
        return result
    
    def same_voice(audio1, audio2):
        nonlocal voice_model
        if audio1 and audio2:
            waveform1 = to_waveform(audio1)
            waveform2 = to_waveform(audio2)
            prediction = voice_model(waveform1, waveform2)
            if prediction >= 0.5:
                result = "✓"
            else:
                result = "X"
        else:
            result = '-'
        return result
    
    # Define a function to handle the audio inputs
    def process_audio(audio1, audio2, audio3, audio4):
        num1 = recognize_number(audio1)
        num2 = recognize_number(audio2)
        num3 = recognize_number(audio3)
        num4 = recognize_number(audio4)
        # This is just a placeholder function. You can add your own logic here.
        return f"{num1}{num2}{num3}{num4}"


    # Define a function to handle the audio inputs
    def verify_audio(audio1, audio2, audio3, audio4, audio5, audio6, audio7, audio8):
        num1 = recognize_number(audio1)
        num2 = recognize_number(audio2)
        num3 = recognize_number(audio3)
        num4 = recognize_number(audio4)

        num5 = recognize_number(audio5)
        num6 = recognize_number(audio6)
        num7 = recognize_number(audio7)
        num8 = recognize_number(audio8)

        verdict1 = same_voice(audio1, audio5)
        verdict2 = same_voice(audio2, audio6)
        verdict3 = same_voice(audio3, audio7)
        verdict4 = same_voice(audio4, audio8)

        # This is just a placeholder function. You can add your own logic here.
        return f"{num5}{num6}{num7}{num8}. Same voice: {verdict1}{verdict2}{verdict3}{verdict4}"
    # Create the Gradio interface
    with gr.Blocks() as demo:
        with gr.Row():
            audio1 = gr.Audio(label="Audio Input 1", type="numpy")
            audio2 = gr.Audio(label="Audio Input 2", type="numpy")
            audio3 = gr.Audio(label="Audio Input 3", type="numpy")
            audio4 = gr.Audio(label="Audio Input 4", type="numpy")
        
        output_text = gr.Textbox(label="PIN")

        # Connect the inputs to the function
        gr.Button("Save").click(
            process_audio, 
            inputs=[audio1, audio2, audio3, audio4], 
            outputs=output_text
        )

        with gr.Row():
            audio5 = gr.Audio(label="Audio Input 5", type="numpy")
            audio6 = gr.Audio(label="Audio Input 6", type="numpy")
            audio7 = gr.Audio(label="Audio Input 7", type="numpy")
            audio8 = gr.Audio(label="Audio Input 8", type="numpy")

        output_text = gr.Textbox(label="Verdict")

        # Connect the inputs to the function
        gr.Button("Verify").click(
            verify_audio, 
            inputs=[audio1, audio2, audio3, audio4, audio5, audio6, audio7, audio8], 
            outputs=output_text
        )
    # Launch the demo
    demo.launch()

server()

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
