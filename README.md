# X-ray Classifier

This project trains a binary classifier to label chest X-rays as `Normal` or `Abnormal`.

Quick commands

Create and activate venv (already created in this workspace):

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
./venv/bin/pip install -r requirements.txt
```

Train model (creates dataset splits under `dataset_split/`):

```bash
./venv/bin/python train.py
```

Evaluate model on test split and save confusion matrix:

```bash
./venv/bin/python evaluate.py
```

Run web app:

```bash
./venv/bin/python app.py
```
