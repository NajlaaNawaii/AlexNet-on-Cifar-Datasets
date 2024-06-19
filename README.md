# Adapting AlexNet for CIFAR-10 and CIFAR-100

This repository contains experiments adapting the AlexNet architecture for the CIFAR-10 and CIFAR-100 datasets. The experiments focus on optimizing the network architecture and data augmentation techniques to achieve better performance on these datasets.

## Experiments Overview

### Performance Comparison of Augmentation Techniques

| Datasets | Top-1 Errors | Top-5 Errors | Accuracy |
|----------|---------------|--------------|----------|
|          | CutOut | MixUp | CutMix | CutOut | MixUp | CutMix | CutOut | MixUp | CutMix |
| CIFAR-10 | 14.78% | 13.12% | 14.90% | 0.90% | 1.00% | 0.81% | 85.39% | 87.27% | 85.10% | 
| CIFAR-100| 44.15% | 41.04% | 53.43% | 15.86% | 15.78% | 23.28% | 56.37% | 58.98% | 46.57% |

*Table 2: Comparisons of MixUp, CutOut and CutMix error rates*

Based on the summary in Table 2 on the model evaluation when trained using the CutOut, CutMix, and MixUp techniques, we notice a significant change in the Top-1 error rate when using the MixUp data augmentation. Our models achieved 13.12% and 41.04% Top-1 error rates on CIFAR-10 and CIFAR-100, respectively, as opposed to 14.78% and 44.15% Top-1 error rates on CIFAR-10 and CIFAR-100, respectively, when trained without MixUp.

CutMix data augmentation did not perform well on CIFAR-100 as we observed an increase in the error rates to 53.43% and 23.28% for Top-1 and Top-5, respectively. In general, MixUp, CutOut, and CutMix techniques performed better in terms of accuracy and error rates on CIFAR-10 compared to the CIFAR-100 dataset.



# Investigating the Effects of Augmentation Techniques on CIFAR-10/100 Images

This repository contains experiments adapting the AlexNet architecture for the CIFAR-10 and CIFAR-100 datasets. The experiments focus on optimizing the network architecture and data augmentation techniques to achieve better performance on these datasets.

## Experiments Overview

### Experiment 1: Adapting AlexNet with Original ImageNet Augmentations
We adapted AlexNet for CIFAR-10 and CIFAR-100 by adjusting kernel sizes to 3x3 and adding an adaptive average pooling layer for 32x32 image resolution. Original AlexNet augmentations (random cropping, horizontal flipping, PCA color augmentation) were used. However, the strong PCA augmentation and random cropping with padding led to unlearnable features and mostly black images, as shown in the figures below. This significantly affected training.

### Experiment 2: Refining PCA Color Augmentation
We refined the PCA color augmentation by clipping pixel values and scaling down perturbation (alpha_std=0.1). Despite these changes, the images remained mostly unrecognizable, similar to Experiment 1.

### Experiment 3: Baseline Training Without Augmentation
We trained the model without any augmentation to establish a baseline. The model achieved high training accuracy (99.47%) but significant overfitting was evident from the low validation accuracy (62.96%) on CIFAR-100. CIFAR-10 showed similar trends with a training accuracy of 82.09% and validation accuracy of 83.53%.

### Experiment 4: Regularization using Basic Augmentation Techniques
To address overfitting, we applied basic augmentation techniques (random horizontal flips, rotations, cropping, and color jitter). Despite these efforts, the model did not perform well, particularly on CIFAR-100. Training accuracies were 52.85% and 99.99%, and validation accuracies were 51.56% and 87.01% for CIFAR-100 and CIFAR-10 respectively.


Another attempt using only horizontal flipping and rotation resulted in training accuracies of 70.69% and 90.41%, and validation accuracies of 51.92% and 83.42% for CIFAR-100 and CIFAR-10 respectively.


## Results Summary

| Datasets   | Experiment 3 (No Augmentation)  | Experiment 4 (Horizontal Flips, Rotations, Cropping, Color Jitter) | Experiment 4 (Horizontal Flipping and Rotation) |
|------------|----------------------------------|---------------------------------------------------------------------|--------------------------------------------------|
| CIFAR-100  | Train Acc: 99.47% Val Acc: 62.96%| Train Acc: 52.85% Val Acc: 51.56%                                    | Train Acc: 70.69% Val Acc: 51.92%                |
| CIFAR-10   | Train Acc: 82.09% Val Acc: 83.53%| Train Acc: 99.99% Val Acc: 87.01%                                    | Train Acc: 90.41% Val Acc: 83.42%                |

Experiment 3 showed significant overfitting without augmentation. Experiment 4's basic augmentation techniques reduced overfitting but did not significantly improve generalization, particularly on CIFAR-100. Simpler augmentations slightly improved training accuracy but validation accuracy remained low. The effectiveness of augmentation techniques varies by dataset complexity.

## Conclusion
The experiments highlight the challenges of adapting AlexNet to CIFAR datasets. Strong augmentations like PCA can hinder performance on low-resolution images. Basic augmentations can reduce overfitting but might not always improve generalization. Further work is needed to find the optimal balance of augmentation techniques for these datasets.
