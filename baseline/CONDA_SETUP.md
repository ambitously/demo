# Conda Setup

This project uses Python 3.8.

## Create environment

```bash
conda create -n rocobench python=3.8 -y
conda activate rocobench
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_dialog.py
```

## Notes

- The Qwen-compatible API calls in `prompting/plan_prompter.py` and `prompting/dialog_prompter.py` use a hard-coded `api_key`.
- Replace `sk-xxx` in those files with your real DashScope API key.
