# Generative Compression of Channel Mixing for Separable CNNs on Microcontrollers

This structure Reproducing the results for paper "Generative Compression of Channel Mixing for Separable CNNs on Microcontrollers":

```
tinyml_modularized/
  tinyml/
    __init__.py
    data.py         # dataset loading/downloading
    models.py       # model architectures
    training.py     # training loops, losses, schedulers
    experiments.py  # experiment orchestration
  main.py           # CLI entrypoint
```

## Run
Example:
```
python main.py --entry tinyml.experiments.run_apnea
```

If no `--entry` is provided, the runner attempts to call the first function starting with `run_`, `experiment`, or `main_experiment` it finds.
:

```
tinyml_modularized/
  tinyml/
    __init__.py
    data.py         # dataset loading/downloading
    models.py       # model architectures
    training.py     # training loops, losses, schedulers
    experiments.py  # experiment orchestration
  main.py           # CLI entrypoint
```

## Run
Example:
```
python main.py --entry tinyml.experiments.run_apnea
```

If no `--entry` is provided, the runner attempts to call the first function starting with `run_`, `experiment`, or `main_experiment` it finds.
