# VUB - Variational Upper Bound for the Information Bottleneck Objective


[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/vwxyzjn/cleanrl)



This library implements the methods described in the paper 'Explainable Regularization in DNNs Using the Information Bottleneck'.
The information bottleneck (IB) is an information theoretic approach for machine learning optimization. VUB is an upper bound for the IB objective that is computable and optimizable in stochastic DNN settings. This Library allows the reconstruction of the experiments demonstrated in the paper.

* 📊 WANDB integration is commented out in the code. To enable it please uncomment it and make sure you have the wandb token in your environment arguments.


Link to the paper [JMLR paper](https://add_link).


## Get started

Prerequisites:
* Python >=3.7.1,<3.11
* [Poetry 1.2.1+](https://python-poetry.org)

Begin by installing the dependencies uding poetry:

```bash
git clone https://github.com/hopl1t/vub.git && cd vub
poetry install
```

Proceed to manually download the imagenet dataset to the vub/datasets/imagenet folder. The original experiment was performed over the first 100K images of the imagenet 2012 dataset. To access this dataset you must first sign up for imagenet at [link](https://www.image-net.org/signup.php).
The smaller 50K examples version can be downloaded from Kaggle either by this link: [link](https://www.kaggle.com/datasets/lijiyu/imagenet), or using the Kaggle API: `bash kaggle datasets download -d lijiyu/imagenet`.

After you've downloaded the imagenet data to the /vub/imagenet folder (that should contain the 'train' and 'val' folders with images) you'll need to download the IMDB dataset, pretrained models for both imagenet and IMDB and then use these models to create a logits dataset (a dataset containing the values in the penultimate layer of each base model). This is all done using the prepare_run.py file very simply by:

```bash
cd vub
poetry shell
python src/prepare_run.py
```

After the pretrained models and logits datasets are saved, we proceed to the experiments:

If you are not using `poetry`, you can install CleanRL with `requirements.txt`:

```bash
# core dependencies
pip install -r requirements/requirements.txt

# optional dependencies
pip install -r requirements/requirements-atari.txt
pip install -r requirements/requirements-mujoco.txt
pip install -r requirements/requirements-mujoco_py.txt
pip install -r requirements/requirements-procgen.txt
pip install -r requirements/requirements-envpool.txt
pip install -r requirements/requirements-pettingzoo.txt
pip install -r requirements/requirements-jax.txt
pip install -r requirements/requirements-docs.txt
pip install -r requirements/requirements-cloud.txt
```




## Citing CleanRL

If you use CleanRL in your work, please cite our technical [paper](https://www.jmlr.org/papers/v23/21-1342.html):

```bibtex
@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
```