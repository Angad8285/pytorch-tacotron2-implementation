# Tacotron 2 Text-to-Speech Implementation

This project is a complete Text-to-Speech (TTS) pipeline built from scratch in PyTorch. It implements the **Tacotron 2** model for generating mel spectrograms from text, paired with a pre-trained **HiFi-GAN** vocoder to synthesize the final audio waveform.

## Features

  * **End-to-End Pipeline**: From raw text and audio to a synthesized waveform.
  * **Tacotron 2 Architecture**: A faithful implementation of the spectrogram prediction network described in the [original paper](https://arxiv.org/abs/1712.05884).
  * **Phoneme-Based Input**: Uses a Grapheme-to-Phoneme (G2P) converter for robust handling of English pronunciation.
  * **High-Fidelity Vocoder**: Integrates a pre-trained HiFi-GAN vocoder for fast and high-quality waveform synthesis.
  * **Universal Training Script**: The training script is hardware-agnostic and will automatically use the best available device: NVIDIA GPU (`cuda`), Apple Silicon GPU (`mps`), or `cpu`.

-----

## Tech Stack

  * Python 3
  * PyTorch
  * librosa (for audio processing)
  * g2p-en & NLTK (for phonemization)
  * scipy (for saving audio files)

-----

## Project Structure

```
multi-speaker-tts/
├── data/                    # For storing datasets like LibriSpeech
├── generated_audio/         # Default output directory for generated samples
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Main source code for the project
│   ├── audio.py
│   ├── config.py
│   ├── data_utils.py
│   ├── model.py
│   └── text.py
├── trained_models/          # For storing trained model checkpoints
├── .gitignore
├── inference.py             # Script for generating audio from a trained model
├── prepare_metadata.py      # Script to parse the raw dataset
├── requirements.txt         # Project dependencies
└── train.py                 # The main training script
```

-----

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Angad8285/pytorch-tacotron2-implementation.git
    cd pytorch-tacotron2-implementation
    ```

2.  **Create and activate a Python virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    You need to download the data packages for the phonemizer. Create a temporary Python script named `download_data.py` with the following content and run it once.

    ```python
    import nltk
    nltk.download('cmudict')
    nltk.download('averaged_perceptron_tagger')
    ```

    ```bash
    python download_data.py
    ```

-----

## Usage

### 1\. Data Preparation

First, download a dataset like [LibriSpeech](https://www.openslr.org/12) or [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) and place it in the `data/` directory. Then, run the metadata script to create the manifest.

```bash
python prepare_metadata.py path/to/your/dataset_folder
```

### 2\. Training

To train the model, run the `train.py` script. You must provide the path to your metadata file and a directory to save checkpoints.

```bash
python train.py data/metadata.csv trained_models/ --epochs 100 --batch_size 16
```

### 3\. Inference

To generate audio from a trained model, use the `inference.py` script. You must provide the text to synthesize and the path to a saved checkpoint.

```bash
python inference.py "Hello world, this is a test." --checkpoint trained_models/tacotron2_epoch_100.pth --output_dir generated_audio
```

The output file will be saved as `output_1.wav` in the `generated_audio` folder. Subsequent runs will create `output_2.wav`, and so on.
