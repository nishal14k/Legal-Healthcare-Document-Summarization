Medical and Legal Document Summarizer (T5-LoRA)
This project is an AI tool designed to summarize complex medical research papers and legal documents.
It uses a technique called LoRA (Low-Rank Adaptation) to fine-tune a small, efficient AI model (T5-Small). This allows it to produce high-quality summaries without requiring expensive computer hardware.

Features
Two Domains: Trained on over 140,000 examples from medical (PubMed) and legal (BillSum) datasets.
Fast and Efficient: The model files are small (about 10MB), making it quick to download and run.
Smart Text Cleaning: The code automatically fixes common formatting errors, like missing spaces after punctuation.
Web Interface: Includes a simple web app where you can paste text and get a summary instantly.

Files in this Repository
train_model.py: The script used to train the AI model on the datasets.
app.py: The script that runs the web interface.
requirements.txt: A list of the software libraries needed to run the code.

Setup and Installation

1. Download the Code

Open your terminal or command prompt and run:

git clone [https://github.com/nishal14k/Legal-Healthcare-Document-Summarization
.git](https://github.com/nishal14k/Legal-Healthcare-Document-Summarization
.git)
cd Legal-Healthcare-Document-Summarization

2. Install Requirements

It is best to use a virtual environment. Install the necessary libraries.

3. Get the Model
The training script saves the model to Google Drive by default.
If you have the trained model: Put the model folder (which contains adapter_model.bin and adapter_config.json) inside a folder named model_files in this directory.
If you want to train it yourself: Run the training script as shown below.

How to Run

Training the Model:
To train the model from scratch (Note: this takes about 3-4 hours on a GPU):
python train_model.py

Note: You may need to change the OUTPUT_DIR setting in the script if you are not running this on Google Colab.

Running the Web App

To start the user interface:
python app.py
This will start a local web server. Open that link in your browser to use the summarizer.

Live Demo:
You can try the app online without installing anything here: https://huggingface.co/spaces/nishal14k/medical-legal-summarizer

Contributing
If you have ideas to improve the project, please open an issue or submit a pull request.
