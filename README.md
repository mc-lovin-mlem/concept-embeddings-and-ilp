# Concept Embedding Analysis And ILP

This is the source code for the publication

- Rabold, Johannes, Gesina Schwalbe, and Ute Schmid. 2020.
  “Expressive Explanations of DNNs by Combining Concept Analysis with ILP.”
  In KI 2020: Advances in Artificial Intelligence.
  Lecture Notes in Computer Science. Bamberg, Germany: Springer International Publishing.

The idea is to use inductive logic programming (ILP) to learn an
expressive surrogate model for parts of a DNN in the following
workflow:

1. **Concept analysis:**
   Find representations of expressive semantic
   visual features within the DNN latent space.
   Add masking of the features as additional output to the DNN.
2. **ILP sample selection:**
   Select samples close to the DNN decision boundary.
3. **Background knowledge creation:**
   From the additional output, infer information about feature
   positions and spatial feature relations.
   Use these to build background knowledge annotations for ILP
   training.
4. **Train/evaluate ILP model** on the DNN predictions:
   Obtain an explainable logical surrogate model for the DNN lower part.

This is realized on a toy example given by a generated dataset.
The surrogate model fidelity was evaluated on 3 finetuned DNN models.

## Repository structure
Folder structure (for sub-folder structure consult the corresponding READMEs):
- [script](./script): All scripts you need for executing the
  workflow. See below for instructions.
- [sources](./sources): Libraries used by the scripts.
- [experiments](./experiments): This is the folder in which experiment results are stored.
- [dataset](./dataset): This is where generated images are stored.
  Also, per model, activations and samples selected as ILP training candidates are stored here. 
- [models](./models): This is where finetuned models are stored, as well as prediction tables.
- [paper-tex](./paper-tex): The LaTeX sources for the publication.

## Installation

### Getting the sources
Just use `git clone --recurse-submodules`.
For further details on working with git submodules see below. 

#### Working with git submodules
A quick guide to work with the submodules (for details and further
tweaks see the [git-book](https://git-scm.com/book/en/v2/Git-Tools-Submodules)).
Basics:
- Submodules are complete git repos on their own; just located within
  the parent (not knowing of it)
- Changes in the submodules must be commited, pushed, and pulled
  normally within the submodule
  (some wrapper commands are available to be called from the parent)
- Submodules are tracked in the main repository
  _only via their commit ID or branch_:
  Changes in the submodule can only be added by commiting them,
  and either adding this specific commit or making it the
  latest one in the branch the parent points to


Some standard steps:

Clone parent repo with submodules:
```bash
git clone <repo_wt_submodules>
git submodule update --init --recursive

# or short
git clone --recurse-submodules
```

Pull parent repo with submodules (automatically applies changes to
submodule links):
```bash
git pull --recurse-submodules
```

Pull remote changes in the submodule repo and 
check out the commit/branch the parent points to
(e.g. the latest commit in master):
```bash
git submodule update --remote
```

Adding a submodule:
```bash
git submodule add <link>
git commit
```

### Installation of dependencies
This project requires
- `python 3.6`,
- a working prolog interpreter,
- an installation of the ILP framework
  [Aleph](http://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html#SEC2).

For python, install the required dependencies via pip by
```bash
python -m pip install -r requirements.txt
```
This will also install the dependencies of the submodules.

## Usage
In the following the steps and corresponding scripts are collected
for the pipeline described in the beginning.

### Create the Picasso dataset
The picasso dataset the experiments are based on is a dataset of generated
faces with segmentation masks for mouth, nose, and eyes that follow the
annotation format of the [FASSEG](http://massimomauro.github.io/FASSEG-repository/) dataset. 
The faces are created by adding randomly selected two eyes, a nose, and a mouth
(all crops of real images) at either their "correct" position in a blank face
or at a permuted position.

Inputs that must be provided, e.g. from the original FASSEG dataset:
- in `dataset/fasseg/heads_original`:
  the original images (from which to crop the features)
- in `dataset/fasseg/heads_labels`:
  the corresponding feature RGB masks in FASSEG segmentation format
  (each the same file name as the masked image)
- in `dataset/fasseg/heads_no_face`:
  some blank faces, e.g. remove facial features via copy & paste
  of bare skin onto them in a common image manipulation program
  
Now call the script
```bash
python script/create_picasso_dataset.py
```

### Workflow for concept analysis
#### Preliminaries
Scripts are run under `python3.6`.

When retraining a model:
Remove previously generated activations if the model is retrained 
(hashes only consider model architecture, not the weights).
The activations lie in the `activations` folder next to the dataset root
(currently: `dataset/fasseg/activations`).

#### Concept analysis
This is summarized in the script [script/analysis_and_ilp_samples.py](script/analysis_and_ilp_samples.py)
(the single steps are detailed below).
Call this from project root with the mandatory `--model` argument:
```bash
python script/analysis_and_ilp_samples.py --model "VGG16"
```
To see the available model specifier and other supported arguments,
call the script with `--help`:
```bash
python script/analysis_and_ilp_samples.py --help
```
The following scripts enable to do single steps out of the
`analysis_and_ilp_samples.py` script
(check the settings, esp. selected model, device and paths):
- To finetune the models: `python script/finetune_all_models.py`
- To pick samples for ILP training: `python -m sources.ilp_samples.picking.py`
- To generate concept embedding outputs for the ILP preparation: `python script/generate_ilp_samples.py`

To load the experiment results, use the functions provided in the
`sources/concept_analysis/ilp_samples` module.
Have a look at the demo in
[demo_load_experiment_results.rst](demo_load_experiment_results.rst).

To summarize the results for all models to a plot in the style of the
one used in the paper, use `script/paper_layerwise_results_graphics.py`
(consult its `--help` option for details). 

### Workflow for ILP explanation generation
The input to the ILP model training are the picked and evaluated examples
from a concept analysis run. These are by default put into a sub-folder of
`experiments/ilp_samples/` (named by the experiment run identifier).
For the following, ensure to provide the correct source and root directories
to the python preparation scripts (consult their `--help` option), or simply
provide the folder name of the experiment run using `--exp_name`.

#### Preliminaries
You have to get a running version of the ILP framework
[Aleph](http://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html#SEC2)
as well as a working Prolog environment. Aleph consists of the prolog file `aleph.pl`.

#### Induce symbolic explanations
To create the files for Aleph, two ILP preparation scripts
(`create_annotations.py`, `create_aleph.py`) need to be run first
as described below:

1. Run the script `create_annotations.py` to generate a file that contains
   the part positions of the samples selected for the explanation
   (`exp_name` is the name of the experiment folder of the concept analysis run):
   ```bash
   python script/create_annotations.py --exp_name <your_exp_name>
   ```
2. To generate the files necessary for Aleph, run the script `create_aleph.py`:
    ```bash
    python script/create_aleph.py --exp_name <your_exp_name>
    ```
   You now have all files Aleph needs in the folder `experiments/ilp/<your_exp_name>`.
3. Place the `aleph.pl` file in the `experiments/ilp/<your_exp_name>` folder.
   Move the `run.pl` file there as well if it is not already there.
4. To induce rules that act as explanations for a model, run the script `run.pl`
   with your Prolog interpreter. This will induce the explanation rules.

#### Evaluate your symbolic surrogate model
Along with the Aleph files (`*.b`, `*.n`, `*.n`), the script `create_aleph.py`
generates the file `evaluation.pl` which contains all the extracted background
knowledge of the annotated samples as well as the facts for samples being
positive or negative.
This can be used to evaluate the fidelity of the explanations when you kickoff
the whole pipeline with new test ILP samples and then only use the
`evaluation.pl` file (make sure to clean the folder `experiments/ilp/<your_exp_name>`
or provide a different source folder to the ILP preparation scripts).
Here you can include the earlier induced rule(s) on the top of the file
and then use Prolog to find Evaluation Metrics like Accuracy, Precision, Recall etc.
An example to find all True Positives of your samples would be the Prolog query:

```
setof(F, (face(F), positive(F)), TP).
```

## Contribute or get help
Just contact the maintainers!

## License
See the [LICENSE](./LICENSE) file.
If you find this work useful, please cite our paper:

```bibtex
@inproceedings{rabold_expressive_2020,
  title = {Expressive Explanations of {{DNNs}} by Combining Concept Analysis with {{ILP}}},
  booktitle = {{{KI}} 2020: {{Advances}} in {{Artificial Intelligence}}},
  author = {Rabold, Johannes and Schwalbe, Gesina and Schmid, Ute},
  year = {2020},
  publisher = {{Springer International Publishing}},
  series = {Lecture {{Notes}} in {{Computer Science}}}
}
```
