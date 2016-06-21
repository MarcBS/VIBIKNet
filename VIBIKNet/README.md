# VIBIKNet for VQA

This file shows how to configure the training of a VIBIKNet model. It assumes that the image features have been 
previously extracted as described in the [feature extraction tutorial](../features_extraction/KCNN/README.md).
 
1) Download the [VQA](http://visualqa.org/download.html) dataset and place it under a folder, for instance: `./VQA`.
  
2) Next, set up your path at `data_engine/convert_to_lists.py`:

 ```python
 dataset_path = './VQA/'
 ```
 
3) And run this script:
 ``
 python data_engine/convert_to_lists.py
 ``

4) If you want to use pretrained Glove vectors, you first need to process them. For doing this, use the script 
`utils/pretrain_word_vectors.py`. 

* In this file, you should modify the path variables. For example, assuming that
the vectors are in the file `./VQA/Glove/glove.42B.300d.txt`:

```python
ROOT_PATH = './VQA/'
base_path = ROOT_PATH +'Glove/'
glove_path = base_path + 'glove.42B.300d.txt'
dest_file = 'glove_300'
```

* Next, process the word vectors by running: `python utils/pretrain_word_vectors.py` 

* And point the `GLOVE_VECTORS` variable from `VIBIKNet/config.py` to this `base_path + dest_file + '.npy'`:


```python
GLOVE_VECTORS = DATA_ROOT_PATH + 'Glove/' + 'glove_300.npy'
```

 
5) Next, let's configure the model. The file `VIBIKNet/config.py` contains the configuration options. All the options
 are explained in the comments. See that file for more information. 

6) Once we configured our model, we have to train it. This is achieved by simply running:

``
python VIBIKNet/main.py
``

8) Once the model has been trained, we can use it on new image and questions! For doing this, we must set the variables 
`MODE`, `RELOAD` and `EVAL_ON_SETS` accordingly in the  `VIBIKNet/config.py`. For example, if we want to obtain the 
answers of the _test_ set, using the model obtained at the end of the epoch _5_, we should do:
 
 ```python
MODE = 'sampling'
RELOAD = 5
EVAL_ON_SETS = ['test']
```

9) Finally, we can also visualize the results. Refer to the [this notebook](VIBIKNet/visualize_results.ipynb) for this.
