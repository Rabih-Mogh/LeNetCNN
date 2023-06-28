# LeNet Convolutional Neural Network
Model created with TensorFlow and trained by the mnist dataset.
The model's summary is displayed when the script is executed.
Training weights are saved in the "checkpoints" directory.
The script will re-train the model on execution if the weights are missing.

### Events:
After evaluating the model on the testing dataset, it will choose one sample of said set at random to plot it alongside the model's prediction.

The script will continuously prompt the user to input an image path of a number drawing. Pressing enter without providing anything will result in the default image being used (28by28.jpg).

_**P.S.** In MSPaint erasing and editing the default image with a black colored brush proved to be an acceptable way to test the model manually._