# Before Name-calling: Dynamics and Triggers of Ad Hominem Fallacies in Web Argumentation

Code and data for NAACL 2018 paper "Before Name-calling: Dynamics and Triggers of Ad Hominem Fallacies in Web Argumentation" by Ivan Habernal, Henning Wachsmuth, Iryna Gurevych, and Benno Stein. Paper PDF available at ACL Anthology: http://aclweb.org/anthology/N18-1036

Use the following citation if you use any of the code or data:

```
@InProceedings{Habernal.et.al.2018.NAACL.adhominem,
    title     = {Before Name-calling: Dynamics and Triggers of Ad Hominem
                 Fallacies in Web Argumentation},
    author    = {Habernal, Ivan and Wachsmuth, Henning and
                 Gurevych, Iryna and Stein, Benno},
    publisher = {Association for Computational Linguistics},
    booktitle = {Proceedings of the 2018 Conference of the North American
                 Chapter of the Association for Computational Linguistics:
                 Human Language Technologies, Volume 1 (Long Papers)},
    pages     = {386--396},
    month     = jun,
    year      = {2018},
    address   = {New Orleans, Louisiana}
}
```


* Contact person: Ivan Habernal, habernal@ukp.informatik.tu-darmstadt.de
    * UKP Lab: http://www.ukp.tu-darmstadt.de/
    * TU Darmstadt: http://www.tu-darmstadt.de/

For license information, see LICENSE. This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.


## Data analysis part

Requirements: Python 3.5+, Jupyter notebook

```
$ cd data-analysis 
```

Set-up a local virtual environment:

```
$ virtualenv env --python=python3
$ source env/bin/activate
```

Install praw and Jupyter

```
$ pip install praw==5.4.0
$ pip install jupyter
$ pip install pandas==0.23.0
$ pip install seaborn==0.8.1
```

Run Jupyter notebook from the `data-analysis` directory

```
$ cd data-analysis/
$ jupyter-notebook
```

* Full analysis of Change my View data is in `01-full-cmv-data-analysis.ipynb`. Note that in order to run this notebook, the full scraped CMV dataset is required. It is not part of this project due to its size (500 MB compressed), you can find the archive [here](https://public.ukp.informatik.tu-darmstadt.de/ih/RedditChangeMyView2017/).
* Analysis of Ad hominem arguments and threads is in `02-ad-hominem-analysis.ipynb`. It uses the "pickled" file `data-analysis/ad-hominem-threads-list-dump.pkl` which was created in the previous step and it is available in this repository. Only for the last couple of cells, the full data is required (see the previous point).
* For manual analysis of Ad hominem threads, `03-ad-hominem-content-analysis.ipynb` prints several random threads of length 3 ending in AH.
* Verification of AH labeling by CMV mods can be found in `04-ad-hominem-labeling-verification.ipynb` which performs also negative sampling based on semantic similarity. To perform the sampling, full CMV dataset is required.
* OP quality analysis for manual labeling and experiments is available in `05-op-quality-analysis.ipynb` and `06-stupidity-controversy-op-plots.ipynb`.

## Running experiments

Requirements: Python 3.5, virtualenv

``
$ cd experiments/
``

Install local virtual environment and activate it

```
$ virtualenv env --python=python3
$ source env/bin/activate
```

Install dependencies

Tensorflow 1.4 (using CPU; you might want to use the GPU version if you prefer)

```
$ pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl
```

Other libraries: LDA and scipy (for the LDA model), NLTK (for tokenizing)

```
$ pip install lda==1.0.5 scipy==1.1.0 nltk==3.2.5
```

### Downloading or re-creating `en-top100k.embeddings.pkl` with embeddings

Due to a 100MB limit on GitHub, the file `en-top100k.embeddings.pkl` is not included (but required for the experiments).

* Download (faster): The file is publicly available [here](https://public.ukp.informatik.tu-darmstadt.de/ih/RedditChangeMyView2017/en-top100k.embeddings.pkl.gz).

* Re-create (slower): Download Glove embeddings (`glove.840B.300d.txt.gz`) and run `vocabulary.py`, it will pickle the embeddings present in the top 100k vocabulary (the file `en-top100k.vocabulary.pkl.gz` is included).


### Classification experiments

Unpack `sampled-threads-ah-delta-context3.tar.bz2` into `data/`

```
$ tar -xvf sampled-threads-ah-delta-context3.tar.bz2 -C data/
```

Run `classification_experiments.py`

```
$ python classification_experiments.py
```

You can adjust the paths where the output for visualization is stored by editing the `classificaton experiments.py` file.


#### Visualization of outputs

Run `Visualization.py` to generate latex files. You will need LaTeX installed in order to generate PDF with visualized word weights.

By default, uses the output folds from the `visualization-context3` folder is used.

### Regression experiments

Train the LDA model first by running `LDA_model.py`:

```
$ python LDAModel.py
```

this will create a file called `LDA_model_50t.pkl` in the current directory.

Run `regression_experiments.py`


