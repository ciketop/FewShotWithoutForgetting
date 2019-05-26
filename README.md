# *Dynamic Few-Shot Visual Learning without Forgetting*

### Reference
Code see [this repo](https://github.com/gidariss/FewShotWithoutForgetting)
Paper see [arXiv](https://arxiv.org/abs/1804.09458) 

### Requirements
- **CUDA** **9.0**, **9.2** or **10.0**
- **Python 2.7**
    - pytorch **0.3.1**
      - if you have CUDA 10.0 installed, you need to install pytorch via `pip install torch==0.3.1 -f https://download.pytorch.org/whl/cu100/stable`
    - torchvision **0.2.0**
      - `pip install torchvision==0.2.0 -f https://download.pytorch.org/whl/cu100/stable`
    - other dependencies see [requirement.txt](./requirements.txt)

## Running experiments on mini-ImageNet.
1. download the MiniImagenet dataset from [here](https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE)
2. extra to `<project root>/datasets`

### Training and evaluating our model on Mini-ImageNet.

**(1)** In order to run the 1st training stage of our approach (which trains a recognition model with a cosine-similarity based classifier and a feature extractor with 128 feature channels on its last convolution layer) run the following command:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifier
```
The above command launches the training routine using the configuration file `./config/miniImageNet_Conv128CosineClassifier.py` which is specified by the `--config` argument (i.e., `--config=miniImageNet_Conv128CosineClassifier`). Note that all the experiment configuration files are placed in the [./config](https://github.com/gidariss/FewShotWithoutForgetting/tree/master/config) directory.

**(2)** In order to run the 2nd training state of our approach (which trains the few-shot classification weight generator with attenition based weight inference) run the following commands:
```
# Training the model for the 1-shot case on the training set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN1
# Training the model for the 5-shot case on the training set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN5 
```

**(3)** In order to evaluate the above models run the following commands:
```
# Evaluating the model for the 1-shot case on the test set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN1 --testset
# Evaluating the model for the 5-shot case on the test set of MiniImagenet.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN5 --testset
```

**(4)** In order to train and evaluate our approach with different type of feature extractors (e.g., Conv32, Conv64, or ResNetLike; see our paper for a desciption of those feature extractors) run the following commands:
```
#************************** Feature extractor: Conv32 *****************************
# 1st training stage that trains the cosine-based recognition model.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv32CosineClassifier
# 2nd training stage that trains the few-shot weight generator for the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv32CosineClassifierGenWeightAttN1 
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv32CosineClassifierGenWeightAttN5
# Evaluate the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv32CosineClassifierGenWeightAttN1 --testset
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv32CosineClassifierGenWeightAttN5 --testset

#************************** Feature extractor: Conv64 *****************************
# 1st training stage that trains the cosine-based recognition model.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv64CosineClassifier
# 2nd training stage that trains the few-shot weight generator for the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv64CosineClassifierGenWeightAttN1 
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv64CosineClassifierGenWeightAttN5
# Evaluate the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv64CosineClassifierGenWeightAttN1 --testset
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv64CosineClassifierGenWeightAttN5 --testset

#************************** Feature extractor: ResNetLike *****************************
# 1st training stage that trains the cosine-based recognition model.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_ResNetLikeCosine
# 2nd training stage that trains the few-shot weight generator for the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_ResNetLikeCosineClassifierGenWeightAttN1 
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_ResNetLikeCosineClassifierGenWeightAttN5
# Evaluate the 1-shot and 5-shot models.
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_ResNetLikeCosineClassifierGenWeightAttN1 --testset
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_ResNetLikeCosineClassifierGenWeightAttN5 --testset
```

### Training and evaluating Matching Networks or Prototypical Networks on Mini-ImageNet.

In order to train and evaluate our implementations of Matching Networks[3] and Prototypical Networks[4] run the following commands:
```
# Train and evaluate the matching networks model for the 1-shot case.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128MatchingNetworkN1
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128MatchingNetworkN1 --testset

# Train and evaluate the matching networks model for the 5-shot case.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128MatchingNetworkN5
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128MatchingNetworkN5 --testset

# Train and evaluate the prototypical networks model for the 1-shot case.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128PrototypicalNetworkN1
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128PrototypicalNetworkN1 --testset

# Train and evaluate the prototypical networks model for the 5-shot case.
CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128PrototypicalNetworkN5
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config=miniImageNet_Conv128PrototypicalNetworkN5 --testset
```

## Running experiments on the ImageNet based Low-shot benchmark

Here provide instructions on how to train and evaluate our approach on the ImageNet based low-shot benchmark proposed by Bharath and Girshick [1]. 

**(1)** First, you must download the ImageNet dataset and set in [dataloader.py](https://github.com/gidariss/FewShotWithoutForgetting/blob/master/dataloader.py#L29) the path to where the dataset resides in your machine. We recommend creating a *dataset* directory `mkdir datasets` and placing the downloaded dataset there. 

**(2)** Launch the 1st training stage of our approach by running the following command:
```
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage1.py --config=imagenet_ResNet10CosineClassifier
```
The above command will train the a recognition model with a ResNet10 feature extractor and a cosine similarity based classifier for 100 epochs (which will take around ~120 hours). You can download the already trained by us recognition model from [here](https://mega.nz/#!fw12RApC!RCnaQd-iEdQuMVZYBFAcPOJKxqrV1Q0m1uTGw6xwDio). In that case you should place the model inside the './experiments' directory with the name './experiments/imagenet_ResNet10CosineClassifier'.

**(3)** Extract and save the ResNet10 features (with the model that we trained above) from images of the ImageNet dataset:
```
# Extract features from the validation image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifier --split=val
# Extract features from the training image split of the Imagenet.
CUDA_VISIBLE_DEVICES=0 python lowshot_save_features.py --config=imagenet_ResNet10CosineClassifier --split=train
```

**(4)** Launch the 2st training stage of our approach (which trains the few-shot classification weight generator with attenition based weight inference) by running the following commands:
```
# Training the model for the 1-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN1
# Training the model for the 2-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN2
# Training the model for the 5-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN5
# Training the model for the 10-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN10
# Training the model for the 20-shot.
CUDA_VISIBLE_DEVICES=0 python lowshot_train_stage2.py --config=imagenet_ResNet10CosineClassifierWeightAttN20
```

**(5)** Evaluate the above trained models by running the following commands:
```
# Evaluate the model for the 1-shot the model.
CUDA_VISIBLE_DEVICES=0 python lowshot_evaluate.py --config=imagenet_ResNet10CosineClassifierWeightAttN1 --testset
# Evaluate the model for the 2-shot the model.
CUDA_VISIBLE_DEVICES=0 python lowshot_evaluate.py --config=imagenet_ResNet10CosineClassifierWeightAttN2 --testset
# Evaluate the model for the 5-shot the model.
CUDA_VISIBLE_DEVICES=0 python lowshot_evaluate.py --config=imagenet_ResNet10CosineClassifierWeightAttN5 --testset
# Evaluate the model for the 10-shot the model.
CUDA_VISIBLE_DEVICES=0 python lowshot_evaluate.py --config=imagenet_ResNet10CosineClassifierWeightAttN10 --testset
# Evaluate the model for the 20-shot the model.
CUDA_VISIBLE_DEVICES=0 python lowshot_evaluate.py --config=imagenet_ResNet10CosineClassifierWeightAttN20 --testset
```
