# Monotonic Lipschitz Networks (MLN) Development

## Setup

Note: Set up a [python virtual environment](https://docs.python.org/3/tutorial/venv.html) before installing for a cleaner setup that does not conflict with your existing packages.

```bash
git clone https://github.com/abdelazizhussein/MLN.git
python3 -m pip install -e MLN
```


## Usage example

To run training:

```
python train.py train -t  ./path/to/training/directory -i ./path/to/input/json/file  -m ./path/to/input/yml/model/config 
```
