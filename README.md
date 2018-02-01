# NN4OMTF CLI tools

Command line tools for [nn4omtf](https://github.com/jlysiak/fuw-nn4omtf) package.

## Requirements

* [nn4omtf](https://github.com/jlysiak/fuw-nn4omtf)
* tensorflow >= 1.4
* python 3.6

## Install

Just run `install` script.  
It creates `~/.nn4omtf_cli` directory and exports its path into bash env.

## Description

Here is only short brief. For more info, please, refer to script files.

* `omtfnntool`
  * shows network model summary
  * creates new model based on provided python code

      Python code should contain no args methood `create_nn()` which returns 3-touple:
```python
  def create_nn():
    
    # Your code goes here...

    return in_placeholder, output, pt_classes
```


