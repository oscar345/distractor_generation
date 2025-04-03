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
