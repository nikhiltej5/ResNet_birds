Trained a ResNet architecture with n = 2 as described above on the given dataset. Useed a batch size of 32 and train for 50 epochs. For dataset, r= 25. Use SGD optimizer with initial learning rate of 10−4. Scheduled the learning rate as appropriate. Experimented with different optimizers other than SGD.

Trained data has 30,000 images while validation has 7500 images each image is of different resolution. Reported the following statistics / analysis:
  Accuracy, Micro F1, Macro F1 on Train and Val splits.
  Plot the error, accuracy and F1 (both macro and micro) curves for both train and val data

In our implementation of ResNet in Section 1.1,we replace the Pytorch’s inbuilt Batch Normalization (nn.BatchNorm2d) with the 5 normalization schemes that you implemented above, giving you 5 new variants of the model. Note that normalization is done immediately after the convolution layer. For comparison, removed all normalization from the architecture, giving us a No Normalization (NN) variant as a baseline to compare with. In total, we have 6 new variants (BN, IN, BIN, LN, GN, and NN).
