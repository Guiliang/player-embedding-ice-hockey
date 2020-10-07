# player-embedding-ice-hockey (Updating...)


My code is based on python 2. Check [requirements.txt](./requirements.txt) for a detail introduction to the python packages.
I recommend python virtual environment for building this project.

## Pre-Processing data
This work applies the play-by-play data from [Sportlogiq](https://sportlogiq.com/en/). Please find a detail description of this data from my thesis (published soon).
run [run_preprocess.py](./sport_data_preprocessing/run_preprocess.py) which include game history into training data.
**To run this code, please prepare you own dataset.**

## Model Training
In the following training file should specify:
source_data_dir = '' # you source data (before pre-preprocessing)
data_store_dir = '' # your training data (after pre-preprocessing)

### VaRLEA Training
The implementation of VaRLEA model is [varlea](./nn_structure/varlea.py).
Run [train_varlea_ice_hockey.py](./interface/train_varlea_ice_hockey.py).
### CVRNN Training
The implementation of CVRNN model is [cvrnn](./nn_structure/cvrnn.py).
Run [train_cvrnn_ice_hockey.py](./interface/train_cvrnn_ice_hockey.py).

The model parameters are recorded in [here](./environment_settings).

## Model Testing

### Player Identification
run [validate_player_id](./testing/validate_id_acc/validate_player_id.py)