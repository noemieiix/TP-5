# DL2 — TP5: Adversarial Examples (*Importance of Git*)

Deep Learning II — Université Paris Dauphine-PSL.

The subject is in **[TP5.pdf](TP5.pdf)**. Read it first.

## Getting started

1. Install [Git](https://git-scm.com/downloads) and the [GitHub CLI](https://cli.github.com) `gh`:
   - Windows: `winget install GitHub.cli`
   - macOS: `brew install gh`
   - Linux: `sudo apt install gh` (on older distributions, see [cli.github.com](https://cli.github.com))
2. Log in once (a browser window opens). GitHub does not accept your password for `git push`; after this, no password is needed:
   ```bash
   gh auth login
   ```
3. This repository is a **template**: you cannot push to it. Create your own copy and clone it:
   ```bash
   gh repo create DL2-TP5 --template matteo-st/DL2-TP5 --private
   gh repo clone DL2-TP5
   cd DL2-TP5
   ```
   GitHub needs a few seconds to copy the template: if the clone fails with `couldn't find remote ref`, wait a few seconds and run `gh repo clone DL2-TP5` again.
   Alternatively, click **Use this template → Create a new repository** at the top of this page, then `git clone https://github.com/<your-username>/DL2-TP5.git`.
4. Set up the environment with one of the files in `config/`:
   ```bash
   # conda
   conda env create -f config/environment.yml
   # or pip / virtualenv
   pip install -r config/requirements.txt
   ```

## Files

| File | Content |
|---|---|
| `train.py` | Trains a model on CIFAR-10 |
| `evaluate.py` | Evaluates a trained model under an adversarial attack |
| `run_xp.py` | Runs several evaluations in a row |
| `models.py` | Model definitions (FC, CNN, ResNet18) |
| `attack_utils.py` | FGSM and PGD attacks |
| `utils.py` | Training, evaluation and plotting helpers |
