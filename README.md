<h1>MMBench & Newt</span></h1>

Official anonymized implementation of

[Learning Massively Multitask World Models for Continuous Control](https://newt-world-models.github.io)

(Anonymous Authors)</br>

[[Website]](https://newt-world-models.github.io) [[Paper]](https://openreview.net/forum?id=MPabX9LEds)

----


## MMBench

MMBench contains a total of **200** unique continuous control tasks for training of massively multitask RL policies. The task suite consists of 159 existing tasks proposed in previous work, 22 new tasks and task variants for these existing domains, as well as 19 entirely new arcade-style tasks that we dub *MiniArcade*. MMBench tasks span multiple domains and embodiments, and each task comes with language instructions, demonstrations, and optionally image observations, enabling research on both multitask pretraining, offline-to-online RL, and RL from scratch.

<img src="assets/0.png" width="100%" style="max-width: 640px"><br/>


## Newt

Newt is a language-conditioned multitask world model based on [TD-MPC2](https://www.tdmpc2.com). We train Newt by first pretraining on demonstrations to acquire task-aware representations and action priors, and then jointly optimizing with online interaction across all tasks. To extend TD-MPC2 to the massively multitask online setting, we propose a series of algorithmic improvements including a refined architecture, model-based pretraining on the available demonstrations, additional action supervision in RL policy updates, and a drastically accelerated training pipeline.

<img src="assets/1.png" width="100%" style="max-width: 640px"><br/>

----

## Getting started

We provide a `Dockerfile` for easy installation. You can build the docker image by running

```
cd docker && docker build . -t <user>/newt:1.0.0
```

This docker image contains all dependencies needed for running MMBench and Newt.

----

## Example usage

Agents can trained by running the `train.py` script. Below are some example commands:

```
$ python train.py    # <-- a 20M parameter agent trained on all 200 MMBench tasks
$ python train.py model_size=XL    # <-- a 80M parameter agent
$ python train.py model_size=B task=walker-walk   # <-- a 5M parameter single-task agent
$ python train.py obs=rgb    # <-- a 20M parameter agent trained with state+RGB observations
```

We recommend using default hyperparameters, including the default model size of 20M parameters (`model_size=L`). See `config.py` for a full list of arguments.

----

## Citation

If you find our work useful, please consider citing our paper as follows:

```
@misc{Anonymous2025Newt,
	title={Learning Massively Multitask World Models for Continuous Control},
	author={Anonymous Authors},
	booktitle={Fourteenth International Conference on Learning Representations (Submission)},
	url={https://openreview.net/forum?id=MPabX9LEds},
	year={2025}
}
```

----

## Contributing

You are very welcome to contribute to this project, but please understand that we will not be able to respond to any pull requests or issues while the submission is under review. Feel free to open an issue or pull request if you have any suggestions or bug reports, but please review our [guidelines](CONTRIBUTING.md) first.

----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
