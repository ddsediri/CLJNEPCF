# Install software requirements
Please run the following pip install command:
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
This command has been added to the bash run script in order to set up your environment on the fly. 

We have tested our networks with the following setup:
```
Python 3.7
Ubuntu 20.04.3
CUDA 11.0
PyTorch 1.7.0
```

# Download code + dataset + pre-trained models

Currently, we have released our inference script along with our trained regression network.

The training set for our regression network is the same as of [Pointfilter](https://github.com/dongbo-BUAA-VR/Pointfilter). We will shortly release our full testset along with the rest of the codebase and trained models.

# How to run

## Preliminaries before running inference
The inference script accepts only point clouds in the '.ply' format. Please convert your test point clouds to this format.

**Please make sure the test set point clouds have both point positions and point normals.**
- Our method estimates **unoriented** normals. We use PCA normals as references to flip our estimated unoriented normals to a more consistent direction.
- Thereafter, our estimated normals are used in the LRMA post-processing algorithm to update point positions.
- We use the following neighborhood sizes, for the corresponding noise levels, for the PCA normals:
* sigma = 0.6%: **60**
* sigma = 0.8%: **150**
* sigma = 1.1%: **200**
* sigma = 1.5%: **200**
* sigma = 2.0%: **200**

## Dataset
Please place the Test directory within the ```Dataset``` directory such that the paths for the Test set is ```Dataset/Test```.

## Run inference only
To run inference, please run the following bash command within the root directory:
```
./Run_Inference_Only.sh
```
Here, the pre-trained regressor available at ```./RegressorPreTrained``` will be used. Also, you may use the following command:
```
python ./Inference.py --checkpoint_path="./RegressorPreTrained/chkpt_cbs_512_ep30_a0.90_b0.01_d0.30_g12.pth.tar" --shapes_list_file="test.txt" --eval_iter_nums=4
```

## Important points
Please note the following:
- Our method removes extreme outliers from the filtered point cloud. This is especially important for high noise point clouds where multiple such outliers may exist. 
- Therefore, the corresponding point, in the ground truth point cloud should also be removed. This corrected ground truth should be used when calculating Angular Error or Point to Surface values for each point.
- To save corrected GTs, set **save_corrected_gt_points** to **True** in line 75 of inference.py and specify the root directory **gt_root** for the original ground truth point clouds in line 68.