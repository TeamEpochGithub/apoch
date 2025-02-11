# Configuration files

The files in this directory are used to control what should be executed and how it should be executed. The most important structural aspects will be explained below.

## Script configuration

At the time of writing there are 3 main scripts, 'train', 'submit' and 'cv'. Each of these has a respective yaml file of the same name to configure the execution.

### Train & CV

Since many of the parameters are the same they will be tackled in the same topic. Parameters should be in this config and not in the model config if they can't be the same for both submission and training. These would be things like train-test split and what scoring method to use.

#### Defaults:
- Name of config: Name specified for checking purposes 'base_train' or 'base_cv'
- Logging config: (hydra/logging) Specifies format of logging messages
- Pipeline: Model or ensemble pipeline to run as 'model: <config>' or 'ensemble: <config>'
- Wandb configuration: Adjustable parameters for interacting with wandb

#### Paths:
These can be whatever is necessary for the implementation. In the base implementation the raw, processed and cache path are included. Recommended would be to have at least the paths for where to read the initial data and where to store the final output

#### Splitting configuration:
- Test size: Determines the split of train, test, validation
- Splitter: Takes a class to split the data in a certain way with custom parameters. These can be taken directly from sklearn or other packages but custom implementations can also be created.

#### Scoring:
Here a class is defined that provides the scoring mechanism. All classes added here should extend the default Scorer class so they can easily be interchanged

#### Multiple instances:
This is an interesting parameter that specifies whether two runs can be done simultaneously on the same machine. The feature was not necessarily created to prevent two runs being done at the same time but to allow for the queueing of multiple runs rather than having to start a sweep. If turned off you can execute the command in multiple terminal windows and they will only execute once the previous is done due to the existence of a lock file.

#### CV specific parameters:
There are a few parameters specific to CV. Since they are mainly dependent on the splitter used there are not many defaults. One that exists is the 'save_folds' param which decides whether to save the models and caches of each fold. This is something that you might want to do before making submissions but not when sweeping over a large amount of parameters such that you run out of storage on your local machine.

#### Other parameters:
Throughout a competition if other parameters are deemed necessary they can easily be added to this config. Keep in mind that they will not affect the hash so any change here will not show up in the hashes. 

### Submit

This is a smaller configuration file that specifies a few parameters. It is used to configure how the submission will be made and what it consists of. 

#### Defaults:
- Name of the config file: 'base_submit'
- Logging config: (hydra/logging) Specifies format of logging messages
- Pipeline: Model or ensemble pipeline to run as 'model: <config>' or 'ensemble: <config>'

#### Paths:
The default paths for this would be where to read the raw data and where to store the pipeline output in submission format. The result path can be dependent on whether you have to manually submit the submissions or whether it is done via notebook format as done on kaggle sometimes.

## Sweep
The wandb configurations for the sweeps should be put in this directory. The parameters set in the sweep run will override those specified in the configuration of the script configuration. As an example if you are sweeping using train.py with a model configured with 256 hidden dims and in the sweep you have a range of 10-300, the hidden_dim param will be overridden with a new value by wandb for every new run. This way of configuring sweeping was chosen as it makes it very easy to sweep over every single possible parameter and sweeps can easily be restarted. More information on how to sweep using wandb will be in the read me.

## Model / Ensemble
The pipeline configurations are put in these directories. The default pipeline is already configured using the modules from training and transformation and all that needs to be done is create a new yaml under model or ensemble to specify the specific blocks. More information on how x_sys, y_sys, train_sys, pred_sys and label_sys works will be found under the documentation of epochlib. For a basic model pipeline configuration using an mlp refer to the 'mlp.yaml' file.

