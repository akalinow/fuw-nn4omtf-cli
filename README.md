# NN4OMTF CLI

Command line tools for [nn4omtf](https://github.com/jlysiak/fuw-nn4omtf) package.

## Requirements

* [nn4omtf](https://github.com/jlysiak/fuw-nn4omtf)
* tensorflow >= 1.4
* python 3.6

* ROOT, root_numpy if converting ROOT dataset to numpy

## Install

Just run `install` script.  
It creates `~/.nn4omtf_cli` directory and exports its path into bash env.
Install required packages on your own - wherever you want.

## Description

Here is only short brief. For more info, please, refer to script files.

* `omtfnntool`
  * shows network model summary
  * creates new model based on provided python code

* `omtfdatasettool`
  * shows OMTFDataset summary
  * creates new dataset from existing `*.npz` files
  * adds new examples from `*.npz` files into existing dataset
  * converts ROOT dictionary dataset files into `*.npz` files
    * it's assumed that ROOT file has specific format and it was produced by OMTF simulator

* `omtfrunner`
  * model training
  * comparing different models
  * gethering statistics

## Creating models

You can create own models by providing builder function as `OMTFNN` constructor argument.
Using `omtfnntool` builder should be in `*.py` file. 
This code should contain no-arg method `create_nn()` which returns 4-touple with:

- input tensor (placeholder), complatible with declared input type
- output tensor, compatible with declared number of classes (#bins + 1)
- pt classes bins edges
- input type constant

This is sample code which builds 3-layer network.

```python
from nn4omtf import utils
from nn4omtf.dataset.const import HITS_TYPE

def create_nn():
    """Create NN which try to classify pt into categories described by arr."""
    arr = [10, 20, 30, 40]
    FL = 10
    SL = 10
    x = tf.placeholder(tf.float32, [None, 18, 2])
    x_in = tf.reshape(x, [-1, 36])
    with tf.name_scope("fc1"):
        # First layer, fully-connected
        W_fc1 = utils.weight_variable([36, FL])
        b_fc1 = utils.bias_variable([FL])
        h_fc1 = tf.nn.relu(tf.matmul(x_in, W_fc1) + b_fc1)
        utils.add_summary(W_fc1, add_hist=False) # Add summary without histogram

    with tf.name_scope("fc2"):
        # Second layer, fully-connected
        W_fc2 = utils.weight_variable([FL, SL])
        b_fc2 = utils.bias_variable([SL])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        utils.add_summary(W_fc2, add_hist=False) # Add summary without histogram

    # Map the features on classes
    with tf.name_scope('fc3'):
        W_fc3 = utils.weight_variable([SL, 5])
        b_fc3 = utils.bias_variable([5])
        y = tf.matmul(h_fc2, W_fc3) + b_fc3
        utils.add_summary(W_fc3) # Add full summary
    
    return x, y, arr, HITS_TYPE.REDUCED

```

## TODO

- `omtfrunner`
    - learning rate param
    - accuracy ival param
    - accuracy dataset size param
    - testing and comparing models

