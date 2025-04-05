# <TITLE FOR PROJECT>

## Installation
For the code to be run on your machine you will need to have the following:

- Python 3.11 or higher
- Cuda 12.1 or higher
- Access to the Llama 3.2 model
- Access to the Mistal model

To make the code run on Habrok and have access to Python 3.11 and Cuda 12.1, you run the following command:

```
module load Python/3.11.5-GCCcore-13.2.0 CUDA/12.1.1
```

With that setup, we can focus on the Python dependencies. To only have the correct versions of the Python dependencies, create a virtual environment and install the dependencies in that environment:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To get access to the Llama and Mistral models, you will have to request access on the [Huggingface](https://huggingface.co/) website. Then to use them on your machine use the `huggingface-cli` command to login. You can follow the instruction given by the command.

```
huggingface-cli login
```

With all things in place, we are now able to run the code.

## Usage
To run the code for preprocessing, training, evaluating, and predicting you can run the `src/main.py` file. This file contains an argument parser which let you run the correct code. An example for this is:

```
python src/main.py --model bert_encoder --mode predict
```

The options for model are:
- `bert_encoder`
- `llama`
- `mistral`
- `baseline`

And the options for mode are:
- `preprocess`
- `eval`
- `train`
- `predict`

### Preprocessing data

Processing the data is done in two ways depending on the model. The BERT decoder model has different preprocessing than the other models, which are decoder only. The preprocess dataset is stored in the `project/data/<model_type>` directory. Where `model_type` is either `encoder_decoder` for the BERT model and `decoder` for the others.

Here is an example of a preprocess command (since you create a `decoder` version of the dataset, you wont have to run this command for the `mistral` and `baseline` models):

```
python src/main.py --model llama --mode preprocess
```

### Training

You can train all the models except the `baseline` model, which will do zero-shot predictions. Both decoder models will be trained to predict all three predictions at once. The BERT model will train to predict one distractor at once. To generate the distractor for the BERT model, we use an additional not yet trained decoder model. This model will be pretrained on the support, question and correct answers, and then the entire model will be trained. Keep this is mind with the time to train the model. To train one of the models just run:

```
python src/main.py --model bert_decoder --mode train
```

### Evaluation

### Predicting

The model will be loaded from disk and the test split from the original `sciq` dataset is loaded. Then the model is run on this dataset and the distractors are created. Those generated distractors will **replace** the original distractors. This version of the dataset is then saved in the `project/predictions/<model>` directory. You can make predictions with a model like so:

```
python src/main.py --model bert_decoder --mode predict
```

## Disclaimer
While this is roughly the same code as we used to train the model, it has been modified so the four actions (preprocess, train, evaluate and predict) can be run independently (so not as one script). To find the original code we used, you can open the [notebook directory](./notebooks). These are marimo notebooks, which can also be run as normal Python scripts.
