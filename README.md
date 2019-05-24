# Machine Learning Engineer Nanodegree
# Convolutional Neural Networks
## Project: Write an Algorithm for a Dog Identification App

### Problem Summary:
- Build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.
- Detect humans face: <br/> **Haar Feature-based Cascades Classifiers** by OpenCV’s implement and **HoG Face Detector** implemented by Dlib.
- Detect dogs: Use a **pre-trained ResNet-50 model** to detect dogs in images. The model come with weights that have been trained on imageNet. 
- **CNN model to classify dog breeds (from scratch)**: <br/> 
Used 16 filters, 32 filters and 64 filters with 2x2 size for the **convolutional layers** and **relu activation function** to extract the information from the images. Those filters are able to collect the regional information of the image and capture lines, curves and even shape appear on the image. The **Max Pooling layers** along with convolutional layers is used to progressively reduce the spatial size of the representation to reduce the number of parameters and computation in the network. The **dropout** is also used to prevent the network from overfitting. After all the convolutional layers, a regular neural network Dense layer with 250 notes is used. In the end the **Softmax function** is used with n notes so that the network is able to calculate the probabilities of all n categories that this image belongs to.
- **CNN model to classify dog breeds (Transfer Learning)**:<br/>
The model uses the **pre-trained VGG-16, VGG-19, RestNet-50, Inception and Xception model** as a fixed feature extractor (**Bottleneck Features**), where the last convolutional output of those pre-trained models are fed as input to our model respectively. A global average pooling layer and a fully connected layer were added, where the latter contains one node for each dog category and is equipped with a Softmax function. In the end we compare the result from each model.
- Run on **TensorFlow** with GPU support
