# Week 2

## Installing Conda Modules
        conda create --name <env_name> --file requirements.txt

## Installing Pip Modules
        pip install -r requirements_pip.txt

## How to run the code
    To run the code (by default will run Fast RCNN model):
        python week2.py 
    
    To change the model to Mask R-CNN it can be done by specifying:
        python week2.py --model "mask_rcnn"

    To run only inference (not training), you can do it by running the following command:
        python week2.py --inference --weights <file_with_weights>
    
    To specify the output directory you can do it by putting the flag --output_dir:
        python week2.py --output_dir <output_directory>

