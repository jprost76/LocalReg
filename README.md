# Learning local regularization for variationnal image restoration

This code is the official implementation of our paper [Learning local regularization for variationnal image restoration](https://arxiv.org/abs/2102.06155)

## Getting started
This project has been developped on python 3.8.5.
After cloning the repository, go on the project directory and start a virtual environement
```
python3 -m venv env_alr
source env_alr/bin/activate
pip install -r requirements.txt
```

## Image restoration
### denoising
```
python3 denoising.py inputs/castle.png --add_noise --std 25
```
Optional flags are :
- --add_noise : add gaussian white noise th the input image. If not specified, the input image is considered as already noisy
- --std std : noise standard deviation (default : 25)
- --model_dir MODEL_DIR name of the trained regularizer directory in regularizers/models_zoo (default : cnn15)
- --out OUT             path to save restored image (if not specified a default path is created in results/ folder)
-  --save_intermediate   save intermediate iterations
-  -l L                  regularization parameter (default=0.1)
-  --oracle              initialize the optimization with the ground truth

### deblurring
```
python3 deblurring.py inputs/castle.png
```
Optional flags :
- --add_noise : add gaussian white noise th the input image. If not specified, the input image is considered as already noisy
- --std std : noise standard deviation (default : 7)
- --model_dir MODEL_DIR name of the trained regularizer directory in regularizers/models_zoo (default : cnn15)
- --out OUT             path to save restored image (if not specified a default path is created in results/ folder)
-  --save_intermediate   save intermediate iterations
- --kernel blurring kernel number (int). kernel are stored in kernels directory
-  -l L                  regularization parameter (default=0.1)
-  --oracle              initialize the optimization with the ground truth