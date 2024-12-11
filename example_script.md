This file contains the scripts for benchmark tasks used in the article.

1. To generate preimage under-approximation for the Cartpole task, run
```
python preimage_main.py --config preimg_configs/cartpole.yaml
```
Flip the boolean arguments for `under_approx` and `over_approx` in the YAML file for generating preimage over-approximation. 

The `threshold` should be greater than 1 for over-approximation and less than 1 for uncer-approximation.