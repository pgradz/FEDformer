This is a fork of https://github.com/MAZiqing/FEDformer with changes to traing FEDformer and Autoformer to crypto trading

## Get Started

1. Install requirements.txt
2. Install https://github.com/pgradz/ml_investing_wne

This repo comes with examplary data for ETH. Other data can be obtained with ml_investing_wne repo.

Run bash scripts to train FEDformer and Autoformer in two modes: classification for the next bar, regression combined with triple barrier method

```bash
bash ./scripts/run_crypto_classifier.sh
bash ./scripts/run_crypto_triple_barrier.sh
```
