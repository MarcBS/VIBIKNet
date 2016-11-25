## Dependencies

- Matlab 2014a or newer.

- Caffe. Download into ~/code/caffe folder or change path in extractCNNfeatures.m
https://github.com/BVLC/caffe

- EdgeBoxes object detection. Download into /repository_root/edges folder.
https://github.com/pdollar/edges

- Piotr Dollar toolbox. Download into /repository_root/piotr_toolbox folder.
https://github.com/pdollar/toolbox

- Inria's Yael library for Matlab. Download into /repository_root/yael folder and compile for you system. See yael's release notes for more information.
https://gforge.inria.fr/projects/yael/

## Steps to follow when executing the KCNN functions

  - 1) extractObjectProposals.m
  
  Extracts object candidates using EdgeBoxes for all the images in the dataset.

  - 2) extractCNNfeatures.m
  
  Extracts ImageNet and Places features for all the images in the datasets. 
The features are extracted on each of the object proposals and their rotations.
In all the following functions ImageNet and Places features will always be
treated separately.

  - 3) applyNormPCA.m	**training only**
  
  Normalizes the features and applies Principal Components Analysis on the training 
samples and stores the parameters.

  - 4) applyNormPCA_test.m
  
  Applies the parameters extracted in applyNormPCA.m to all the samples in the dataset.

  - 5) learnGMM.m	**training only**
  
  A Gaussian Mixture Model is learned using the PCA features of the training samples
and stores the corresponding parameters.

  - 6) transformFVs.m
  
  Calculates the fisher vectors for each image in the dataset using the parameters learned
in learnGMM.m

  - 7) applyNormPCAFV.m	**training only**
  
  Again, normalizes the features and applies PCA on the training samples after the FVs 
calculation.

  - 8) applyNormPCAFV_split.m
  
  Uses the parameters learned in the previous function for reducing the dimensionality 
of the FVs for all the samples. Stores the samples in separate data splits 
{train,val,test}.


## General KCNN extraction scheme

![KCNN_scheme](../../docs/CVPR_KCNN.png)


## Citations

```
Liu Z. 
Kernelized Deep Convolutional Neural Network for Describing Complex Images. 
arXiv preprint arXiv:1509.04581. 2015 Sep 15.
```

```
Zitnick CL, Dollár P. 
Edge boxes: Locating object proposals from edges. 
In European Conference on Computer Vision 2014 Sep 6 (pp. 391-405). Springer International Publishing.

```
