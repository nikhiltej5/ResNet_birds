Trained a ResNet architecture with n = 2 as described above on the given dataset. Used a batch size of 32 and trained for 50 epochs. For the dataset, r = 25. Used the SGD optimizer with an initial learning rate of 10^-4. Scheduled the learning rate as appropriate. Experimented with different optimizers other than SGD.

The trained data has 30,000 images, while the validation set has 7500 images, each of a different resolution. Reported the following statistics / analysis:
- Accuracy, Micro F1, Macro F1 on Train and Val splits.
- Plot the error, accuracy, and F1 (both macro and micro) curves for both train and val data.

In our implementation of ResNet in Section 1.1, we replaced PyTorch’s inbuilt Batch Normalization (`nn.BatchNorm2d`) with the 5 normalization schemes that you implemented above, giving us 5 new variants of the model. Note that normalization is done immediately after the convolution layer. For comparison, we removed all normalization from the architecture, giving us a No Normalization (NN) variant as a baseline to compare with. In total, we have 6 new variants (BN, IN, BIN, LN, GN, and NN).
