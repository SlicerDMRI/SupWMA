# SupWMA

This repository releases the source code, pre-trained model and testing sample for the work, "SupWMA: consistent and efficient tractography parcellation of superficial white matter with deep learning," which is accepted by the ISBI 2022 (finalist for best paper award).

Compared to several state-of-the-art methods, SupWMA obtains a highly consistent and accurate SWM parcellation result. In addition, the computational speed of SupWMA is much faster than other methods.

![v4_SupWMA_Overview](https://user-images.githubusercontent.com/56477109/150537721-9619c9f6-98f0-4a02-ae4f-4794b99235fd.png)

## License

The contents of this repository are released under an [Slicer](LICENSE) license.

## Dependencies:

  `conda create --name SupWMA python=3.6.10`
  
  `conda activate SupWMA`
  
  `pip install conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch`
  
  `pip install git+https://github.com/SlicerDMRI/whitematteranalysis.git`
  
  `pip install h5py`
  
  `pip install sklearn`

## Train without contrastive learning
Train with our dataset (available upon request)
1. Download `TrainData.zip`  and `tar -xzvf TrainData.zip`
2. Run `sh train_supwma_without_supcon.sh`

## Train with contrastive learning
Train with our dataset (available upon request)
1. Download `TrainData.zip` and `tar -xzvf TrainData.zip`
2. Run `sh train_supwma_with_supcon.sh`

## Train using your custom dataset
Your input streamline features should have size of (number_streamlines, number_points_per_streamline, 3), and size of labels is (number_streamlines, ). You can save/load features and labels using .h5 files.

Although we have swm outliers class and other (dwm) class in our dataset, your dataset is not required to have these classes. For example, you can train your model using a dataset with 600 swm cluster classes. The training process will be the same. 

It is recommended to start training your custom dataset without contrastive learning, which is easier for debugging. 

If you have already obtained reasonably good results, then you can use contrastive learning to boost your performance.

## Train/Val results
We calculated the accuracy, precision, recall and f1 on 198 swm clusters and one "non-swm" cluster (199 classes). One "non-swm" cluster consists of swm outlier clusters and others (dwm).

## Test (SWM parcellation)
1. Install 3D Slicer (https://www.slicer.org) and SlicerDMRI (http://dmri.slicer.org).
2. Download `TrainedModels.zip` (https://github.com/SlicerDMRI/SupWMA/releases) to `./`, and `tar -xzvf TrainedModel.zip`
3. Download `TestData.zip` (https://github.com/SlicerDMRI/SupWMA/releases) to the `./`, and `tar -xzvf TestData.zip`
4. Run `sh SupWMA.sh`

## Test parcellation Results

Vtp files of 198 superficial white matter clusters and one Non-SWM cluster are in `./SupWMA_parcellation_results/[subject_id]/[subject_id]_prediction_clusters_outlier_removed`. 

You can visualize them using 3D Slicer.

![SWM_results](https://user-images.githubusercontent.com/56477109/150535586-28f30123-5fd1-4a9c-a81e-499d5abfd65d.png)
