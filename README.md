# GPT 2 Finetune

This is an overview of the code for fine tunning a text generation model using the GPT-2 architecture and a csv dataset. The code covers the following steps:

## 1. Installing Dependencies

Before using the code, ensure you have the necessary Python packages installed. You can use the following command to install the required dependencies:

```bash
!pip install transformers datasets torch==2.0.1 accelerate
```

## 2. Import Libraries

The code imports essential libraries and modules required for training and testing, including PyTorch, Hugging Face Transformers, and Hugging Face Datasets.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric
```

## 3. Importing and Processing Dataset

The dataset is loaded from a CSV file (data.csv). The code splits the dataset into training and validation sets, with 90% for training and 10% for validation.

## 4. Tokenization

The input and target sequences are tokenized using the GPT-2 tokenizer. Tokenization involves setting the padding and truncation options for input and target sequences.

## 5. Import Model

The pre-trained GPT-2 model and tokenizer are loaded. The model name "gpt2" is used for both the model and tokenizer.

## 6. Train Model

The training process is configured using the TrainingArguments class. Key training parameters include:
- `output_dir`: The directory where the trained model will be saved.
- `num_train_epochs`: The number of training epochs (0.5 in this example).
- `per_device_train_batch_size`: Batch size for training.
- `evaluation_strategy`: Evaluation strategy (in this case, based on steps).
- `eval_steps`: Frequency of evaluation steps.
- `save_steps`: Frequency of model saving.
- `logging_steps`: Frequency of logging training information.
- `logging_dir`: Directory for storing training logs.

A Trainer is created using the model, training arguments, and training dataset. The model is fine-tuned using the `train` method.

## 7. Save Model

The trained model is saved to the specified output directory.

## 8. Running Inference

Inference can be performed using the trained model. An example input is tokenized, passed through the model for text generation, and the generated output is decoded. The code snippet at the end of the file provides an example of how to run inference with the model.

**Note:** Ensure that you have a GPU available for training and inference to benefit from accelerated performance when using the 'cuda' device.

This code provides a foundation for training and deploying a GPT-2-based model for text generation tasks, particularly for correcting HTML code in the provided example. You can adapt and extend the code to other text generation applications as needed.
