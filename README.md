# Voice ID: Veryfying clients with voice

The project aims to develop a voice-based client verification system for a Revolut digital bank using machine learning to enhance security and authentication. The system verifies client PINs from speech and analyzes vocal patterns using a Convolutional Neural Network (CNN) trained on the Google Command speech dataset, achieving 86% accuracy. It processes audio recordings converted into waveforms to determine whether the voice belongs to a registered client.

![alt text](path/to/image1.png)

## Business understanding

Voice authentication is becoming a highly secure method for user authentication, especially in digital banking, where it helps mitigate security breaches and fraud. By using the unique characteristics of a person’s voice, this technology provides a stronger layer of protection compared to traditional methods like passwords. For example, [HSBC UK saw a 50% reduction](https://www.about.hsbc.co.uk/news-and-media/hsbc-uks-voice-id-prevents-gbp249-million-of-attempted-fraud) in telephone fraud after implementing voice biometrics, demonstrating its effectiveness in reducing fraud and enhancing customer trust.

## Data understanding

The Speech Commands Dataset v0.02 consists of over 105,000 one-second .wav audio files, each containing a single spoken English word from a set of 35 unique commands. Organized into folders by word, the dataset includes recordings from various speakers and is available in the TensorFlow Datasets catalog. Additionally, it features background noise samples and labels like "silence" and "unknown" to improve noise handling.

## Modeling

## Evaluation

## Conclusions

## Repository Navigation

[Final Notebook](/path/to/file)
[Presentation](/path/to/file)

 ## Reproduction instsructions








First Simple Model - CNN
Model is located in src/__main__.py

* Trained on one word 'marvin' for now
* Around 800 rows

Best result at this point:
Train loss: 0.5531. Test Loss: 0.6530. Precision 0.6522. Recall: 0.7792. F1 score: 0.7101. 

