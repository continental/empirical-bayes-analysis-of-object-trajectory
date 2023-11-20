# Continental-Empirical-Bayes-Analysis
Demo for extracting prior knowledge (model parameter distribution) from Waymo Open Motion Dataset v1.1 via Empirical Bayes Analysis.


## Getting started

### Installing
1. Clone this repository
2. Download [Waymo Open Motion Dataset](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)  and put the data into "demo/data" directory
3. Install Dependencies
    1. pip install requirements.txt
    2. install waymo_open_dataset according to [here](https://github.com/waymo-research/waymo-open-dataset/tree/master)

### Running
1. Run **demo/preprocess_waymo.ipynb** as entrypoint.
2. Excute Empirical Bayes Analysis with:
    1. **demo/ego_traj_analyse.ipynb** for ego vehicle trajectory.
    2. **demo/agt_traj_analyse.ipynb** for other object trajectory.
3. Evaluate AIC, BIC and representation error with **demo/result_evaluation.ipynb**.

## Prior knowledge
- We provide the extracted model prior distribution and observation noise distribution for different objects and timescales in **demo/logs/gradient_tape**.
- An example of integrating prior knowledge is provided in **demo/integrate_prior.ipynb**.

## Reference
 1.  [An Empirical Bayes Analysis of Object Trajectory Models](https://arxiv.org/abs/2211.01696)
 

## Documentation

For further help, see the API-documentation or contact the maintainers.

## License

Copyright (c) 2023 Continental Corporation. All rights reserved.