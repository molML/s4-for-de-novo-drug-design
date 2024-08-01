# S4 for De Novo Drug Design

Hello hello! :raising_hand_man: Welcome to the official repository of [Chemical language modeling with structured state space sequence models](https://www.nature.com/articles/s41467-024-50469-9)!

First things first, thanks a lot for your interest in our work and code :pray: Please consider starring :star: the repository if you find it useful &mdash; it helps us know how much maintenance we should do! :innocent:

This document will walk you through the installation and usage of our codebase. By completing this document, you'll be able to pre-train, fine-tune, and sample your own structured state-space sequence model (S4) to design molecules *in only 4 lines of code* :heart_eyes: Let's get started :rocket:


## Installation :hammer_and_wrench:

You first need to download this codebase. You can either click on the green button on the top-right corner of this page and download the codebase as a zip file or clone the repository with the following command, if you have git installed:

```bash
git clone https://github.com/molML/s4-for-de-novo-drug-design.git
```

We'll use `conda` to create a new environment for our codebase. If you haven't used conda before, we recommend you take a look at [this tutorial](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) before moving forward.


Otherwise, fire up a terminal *in the (root) directory of the codebase* and type the following commands:

```bash
conda create -n s4_for_dd python==3.8.11 
conda activate s4_for_dd 
conda install pytorch==1.13.1 pytorch-cuda==11.6 -c pytorch -c nvidia  # install pytorch with CUDA support
conda install --file requirements.txt -c conda-forge  
python -m pip install .  # install this codebase -- make sure that you are in the root directory of the codebase
```

> [!WARNING]
> If you don't have (or need) GPU support for pytorch, replace the third command above with following: `conda install pytorch==1.13.1 -c pytorch`


That's it! You have successfully installed our codebase; `s4dd` to name it. Now, let's see the magical 4 lines to design bioactive molecules with S4 :crystal_ball:


## Designing Molecules with S4 :woman_technologist:
Here we are: we pre-train an S4 on ChEMBL, fine-tune on a set of bioactive molecules for the protein target PKM2, and design new molecules. All with the following 4 lines of code:

```python
from s4dd import S4forDenovoDesign

# Create an S4 model with (almost) the same parameters in the paper.
s4 = S4forDenovoDesign(
    n_max_epochs=1,  # This is for only demonstration purposes. Set this to a (much) higher value for actual training. Default: 400.
    batch_size=64,  # This is also for demonstration purposes. The value in the paper is 2048.
    device="cuda",  # Replace this with "cpu" if you didn't install pytorch with CUDA support.
)
# Pretrain the model on ChEMBL
s4.train(
    training_molecules_path="./datasets/chemblv31/mini_train.zip",  # This a 50K subsample of the ChEMBL training set for quick(er) testing.
    val_molecules_path="./datasets/chemblv31/valid.zip",
)
# Fine-tune the model on bioactive molecules for PKM2
s4.train(
    training_molecules_path="./datasets/pkm2/train.zip",
    val_molecules_path="./datasets/pkm2/valid.zip",
)
# Design new molecules
designs, lls = s4.design_molecules(n_designs=32, batch_size=16, temperature=1.0)
```

Voila! :tada: You have successfully trained your own S4 model from scratch for  *de novo* drug design and designed molecules in 4 lines :nazar_amulet: Examples for each step are also available in the [`examples/`](https://github.com/molML/s4-for-de-novo-drug-design/examples) folder.

> [!WARNING]
> Make sure that you replace the `"cuda"` argument with `"cpu"` if you didn't install pytorch with CUDA support.

> [!IMPORTANT]
> Use a smaller batch size if you face out-of-memory errors.


You can do more with `s4dd`, *e.g.,* save/load models, calculate likelihoods of molecules, and monitor model training. Let's quickly cover those :running:

## Additional Functionalities :joystick:

### 1. Save/Load Models :floppy_disk:

Saving models are useful to resume training later or to design molecules without repeating the training, *e.g.,* for fine-tuning and chemical space exploration. That's why we made model saving in `s4dd` as simple as:

```python
s4.save("./models/foo")  # s4 is the S4 model we trained above.
```

Then to load the same model in another file/session:

```python
# load it back
loaded_s4 = S4forDenovoDesign.from_file("./models/foo")
...  # resume training with `loaded_s4` or design molecules...
```

### 2. Calculate Molecule Likelihoods :game_die:
In addition to designing molecules, S4 (or any chemical language model), can compute likelihoods of molecules, enabling new evaluation perspectives. A detailed discussion of 'how' is available in our paper. 

Let's dive back into the code here and see how we can compute the (log)likelihood of a molecule via `s4dd`:
```python
lls = s4.compute_molecule_loglikelihoods(["CCCc1ccccc1", "CCO"], batch_size=1)
```

As usual, it's that easy! :man_shrugging:


### 3. Monitor Model Training :mag:

Tracking the model training is crucial for any machine learning project. Our codebase, `s4dd`, provides out-of-the-box functionality to help you fellow machine learning researcher :crossed_fingers:

`s4dd` implements four "callbacks" to monitor model training:

 - `EarlyStopping` callback stops the training if an evaluation metric stops improving for a pre-set number of epochs and saves some precious training time :moneybag:
 - `ModelCheckpoint` saves the model per fixed number of epochs so that the intermediate models are available for analysis :microscope:
 - `HistoryLogger` saves the training history at every epoch to monitor the training and validation losses :chart_with_downwards_trend:
 - `DenovoDesign` designs molecules in the end of every epoch with selected temperatures to track model's generation capabilities :pill:

Integrating any of those callbacks to the model training is almost trivial &mdash; you just need to pass them as a list to the `train` method:

```python
from s4dd import S4forDenovoDesign
from s4dd.torch_callbacks import EarlyStopping, ModelCheckpoint, HistoryLogger, DenovoDesign

s4 = S4forDenovoDesign(
    n_max_epochs=10,
    batch_size=32,
    device="cuda", 
)
s4.train(
    training_molecules_path="./datasets/chemblv31/train.zip",
    val_molecules_path="./datasets/chemblv31/valid.zip",
    callbacks=[
        EarlyStopping(
            patience=5, delta=1e-5, criterion="val_loss", mode="min"
        ),
        ModelCheckpoint(
            save_fn=s4.save, save_per_epoch=3, basedir="./models/"
        ),
        HistoryLogger(savedir="./models/"),
        DenovoDesign(
            design_fn=lambda t: s4.design_molecules(
                n_designs=32, batch_size=16, temperature=t
            ),
            basedir="./models/",
            temperatures=[1.0, 1.5, 2.0],
        ),
    ],
)
```


## Documentation :scroll:
Are you interested in doing more with `s4dd`? Or you need more information about some of `s4dd`'s (very cool) functionalities? Then you can find our online documentation useful. [Here](https://molml.github.io/s4-for-de-novo-drug-design/) you can find the detailed description of each single class and function in `s4dd`. Happy reading! :nerd_face:

Or, are you only interested in a deeper look into the results in our work? 🔍 Then, [here](https://zenodo.org/records/11085650) is a link our Zenodo repository 💼

##  Closing Remarks :fireworks: 

Thanks again for finding our code interesting! Please consider starring the repository :sparkles: and citing our work if this codebase has been useful for your research :woman_scientist: :man_scientist: 


```bibtex
@article{ozccelik2024chemical,
  title={Chemical language modeling with structured state space sequence models},
  author={{\"O}z{\c{c}}elik, R{\i}za and de Ruiter, Sarah and Criscuolo, Emanuele and Grisoni, Francesca},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={6176},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

If you have any questions, please don't hesitate to open an issue in this repository. We'll be happy to help :man_dancing: 

Hope to see you around! :wave: :wave: :wave:                                
