This file contains the scripts for benchmark tasks used in the article.

1. To generate preimage under-approximation for the Cartpole task, run
```
python preimage_main.py --config preimg_configs/cartpole.yaml
```
To generate preimage over-approximation, flip the boolean arguments for `under_approx` and `over_approx` in the YAML file. 

The `threshold` should be greater than 1 for over-approximation and less than 1 for under-approximation.

2. To generate preimage under-approximation for the Vehicle Parking task, run
```
python preimage_main.py --config preimg_configs/auto_park_under.yaml
```

Correspondingly, to generate preimage over-approximation for the Vehicle Parking task, run
```
python preimage_main.py --config preimg_configs/auto_park_over.yaml
```

3. To generate preimage approximation for the Collisian Avodiance models, run
```
python preimage_main.py --config preimg_configs/vcas.yaml
```
Change the parmater `vcas_idx` to specify the vcas model ID and set the corresponding model path using the parameter `onnx_path`

4. For the lunarlander task, run
```
python preimage_main.py --config preimg_configs/lunarlander.yaml
```

5. For the dubinsrejoin task, run
```
python preimage_main.py --config preimg_configs/dubinsrejoin.yaml
```

