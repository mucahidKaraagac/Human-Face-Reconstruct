# Human-Face-Reconstruct
This project about a building from the scratch Neural Network. Network customized to recover CelebA dataset images. With this purpose PyTorch framework used for the modeling network. CelebA dataset is a dataset which is stored famous peoples’ portrait images. 
Task is using the align ones for the reconstruct giving images form theirs left hand halves. Project using the encoder-decoder structure type for Neural Network. Encoder-decoder network is basically having 3 portions of it. 
Encode part of it gather any data can be taken on given input. Besides this part also image size reduced due to make a zoom affect to gather more minor information of the giving data.
 After encode part a fully connect part coming in that part there is no stride or pooling (resizing) operation to reduce dimension of input height and weight, so we are gathering a lot of minor detail data on given input Also using conv2d layer instead of fully connected ones for more robustness on the any overfitting issue. 
Decode part of network basically every layer using upsample and conv2d for the extend input height, width and doing an unpack like operation for the construct an output image for the network.  
Constructed output image (Tensor) used on MSELoss for the calculate loss for the optimizing gradients in all the neurons in our network. With this operation our network fine tune it selves help of the loss value and starting learning how to operate the given input.
Optimizer is Adam algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data. It is used the loss values to do its job which is provided by MSELoss.

![image](https://user-images.githubusercontent.com/58566560/172106939-23924575-0b98-432c-beca-29366081d23a.png)
