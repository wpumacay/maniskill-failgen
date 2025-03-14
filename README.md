# maniskill-failgen
A implementation of failgen in maniskill

## How to use

Install dependencies in virtual environment:

```bash
# Create a virtual environment (could use conda as well)
virtualenv venv
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

Install this package in developer mode:

```bash
pip install -e .
```

Run the data collection script (check for flags in the script)

```bash
python examples/ex_failgen_data_collection.py --headless --save-video
```
