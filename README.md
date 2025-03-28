# Racing Simulator

## Dependencies

- Python 3.10.12
- mlagent 1.1.0
- torch 2.2.1

## Installation

You can install pyenv via your package manager. Then install Python version 3.10.12 using pyenv:

```bash
pyenv install 3.10.12
pyenv local 3.10.12
```

Once done, execute:

```bash
source activate_pyenv.sh
```

If you run the `python --version` command, you should see the correct version (3.10.12).

You can create your Python environment and launch it using the following commands:

```bash
python -m venv env

source env/bin/activate
```

You are now in your Python environment (`deactivate` to exit).

Run the following commands to complete your mlagent and dependency installation:

```bash
pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cu121

pip install mlagents==1.1.0

mlagents-learn --help
```