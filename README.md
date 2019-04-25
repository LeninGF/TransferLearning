# Transfer Learning
This script will test transfer learning to use it in personal datasets

## Uses of transfer Learning
- **Fixed Feature Extractor:** we use last flatten vector as feature extractor to be used in a SVM
- **Pre-trained CNN:** only the classifier is trained on the new dataset. This is recommended for small datasets. For instance if ImageNet has 1000 classes, we substitute the last CNN for one where there are as many classes as outputs our dataset has
- **Fine-Tunning:** Initial layers are frozen. The remainining layers are re-trained. It is recommended for Datasets of *Medium Size*