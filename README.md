# octconv with pytorch

------

[PyTorch](https://github.com/d-li14/octconv.pytorch/blob/master/pytorch.org) implementation of Octave Convolution in [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049)



### Abstract

In natural images, we can factorize them into higher frequencies which are encoded with fine detail and lower frequencies which are encoded with global structures. To fully use the information in natural images, the authors design a novel Octave Convolution (Octave Convolution) operation to store and process feature maps that vary spatially “slower” at a lower spatial resolution reducing both memory and computation cost. What’s more, Octave Convolution is a plug-and-play structure and we can use it in different convolution structure easily. By applying the mothed to the experiments, I get the same results with the authors that Octave Convolution perform better than origin network. 

 

### **1.** **Introduction**

Although Convolutional Neural Networks have already got excellent performance on many image classification or prediction tasks, they still have some inherent redundancy in dense model parameters and in the channel dimension of feature maps. Many researches have many experiments to change the structure of the network, however, they ignore the importance of the origin information of the train data. The authors are motivated by a mathematical theory which tells us that natural image can be decomposed into a low and a high spatial frequency part. Different parts have different information to describe the natural image. For example, the low spatial frequency component that describes the smoothly changing structure and a high spatial frequency component that describes the rapidly changing fine details as show in Figure(a). They map these two parts into different size of features maps. So, they can use less parameters to train the low spatial frequency and use more parameters to change the high spatial frequency but there are information communication between them.

![](https://github.com/Vilinz/octconv/blob/master/images/1.png?raw=true)

### **2.** **Method**

To understand the method clearly, we have a simply discussion about spatial frequency. A natural image can be decomposed into a low and a high spatial frequency part as show in Figure(b). The high spatial describes the rapidly changing fine detail and the low spatial describes the smoothly changing structure. similarly, the output feature maps of a convolution layer can also be seen as a mixture of information at different frequency. Since the low spatial contains less information, we can do some compression to reduce spatial redundancy. In Octave Convolution, they do convolution on their own spatial frequency. In addition, there is information exchanging between the low spatial and the high spatial as show in Figure(d).

![img](https://github.com/Vilinz/octconv/blob/master/images/2.png?raw=true)

To achieve the above target, the authors define a new octave convolution to replace the vanilla convolution due to differences in spatial resolution in the octave features, which is expected to not only effectively process the low-frequency and high-frequency parts in their corresponding frequency tensor, but also enable efficient communication between them. Since the size between the high spatial and the low spatial are different, when it comes to the inter-frequency communication, folding the up-sampling over the feature tensor ![img](file:///C:/temp/msohtmlclip1/01/clip_image007.png) or the down-sampling of the feature tensor ![img](file:///C:/temp/msohtmlclip1/01/clip_image009.png) into the convolution using following formulas is necessary. The procedure is as show below.

![img](https://github.com/Vilinz/octconv/blob/master/images/3.png?raw=true)

So, the output ![img](https://github.com/Vilinz/octconv/blob/master/images/9.png?raw=true) of the Octave Convolution using average pooling for down-sampling as:

![img](https://github.com/Vilinz/octconv/blob/master/images/10.png?raw=true)

Octave Convolution is a plug-and-play replacement for the vanilla convolution, so it can be applied to any model without changing the struct of the model.

 

### **3.** **Implementation**

Since the Octave Convolution is formulated as a single, generic, plug-and-play convolutional unit that can be used as a direct replacement of (vanilla) convolutions without any adjustments in the network architecture. I try to apply this change to Resnet50. For a network, I divide the network into three parts. 

Firstly, for a natural image, I must separate it into low-frequency part and high-frequency part. So in the first Bottleneck of the Resnet, I get two inputs by down-sampling and using origin input. After the first convolution by proportion parameter, I get the low-frequency part and high-frequency part.

Secondly, I make convolution in different frequency parts. Since different frequency parts have different tensors size, I cannot communicate two parts directly. So, I make up-sampling or down-sampling to communicate each other.

Thirdly, to get the result we want, I must combine two parts into one part. So, in the last Bottleneck of the Resnet, I add them together after convolution so as to make Linear Connection.

 

### **4.** **Experiment**

The authors use the ImageNet dataset to finish the experiment. However, the dataset is too large for us students. So, I choose CIFAR10 to do the experiments in the paper. Despite the dataset is not that large, the running time is still relatively long. According to the experiments in paper, I do some experiments CIFAR10 with Resnet50. The origin struct of Resnet can get from [here](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py). 

To compare the results with origin Resnet50, I train origin network firstly and apply the same parameters to the Octave Convolution. I train them for the same epochs and draw their losses and accuracies in the same graph to compare them.

As the experiments in the paper, I set different ratios (0, 0.125, 0.25, 0.5, 0.75) of the low spatial. The experiment results I get are showed as follow. 

![img](https://github.com/Vilinz/octconv/blob/master/images/4.png?raw=true)

![img](https://github.com/Vilinz/octconv/blob/master/images/5.png?raw=true)

We can see from the results that Resnet50 with Octave Convolution perform better than origin Resnet50 and when I set the proportion to 0.125, I get the best result in all the experiments I have done, which have the same conclusion as the experiments in the paper.

To verify the importance of the information exchange, I ablate on down-sampling and inter-octave connectivity on CIFAR10 and compare the results. But I change the ratio into 0.125. I find that Octave Convolution performs little better with both Low->High and High->Low information communication. The experiment results I get are as show.

![img](https://github.com/Vilinz/octconv/blob/master/images/6.png?raw=true)

 

| operation                   | Low->High | High->Low | Top-1(%) |
| --------------------------- | --------- | --------- | -------- |
| Oct-ResNet-50   Ratio:0.125 |           |           |          |
|                             | √         |           |          |
|                             |           | √         |          |
|                             | √         | √         |          |

 ![](https://github.com/Vilinz/octconv/blob/master/images/7.png?raw=true)

![](https://github.com/Vilinz/octconv/blob/master/images/8.png?raw=true)

### **5.** **Conclusion**

I examine Octave Convolution on Resnet50 by replacing the regular convolutions with Octave Convolution (except the first convolutional layer before the max pooling). The resulting networks only have one global hyper-parameter α, which denotes the ratio of low frequency part. I find that Octave Convolution performs better than origin network in CIFAR10 dataset. At the same time, I use different ratio of low frequency part to train Octave Convolution. I get the same conclusion with the authors that when I set ratio to 0.125, Octave Convolution has best accuracy among 0.125, 0.25, 0.5, 0.75. What’s more, when I ablate on down-sampling and inter-octave, the accuracy of the model decreases.

 

### **6.** **Source Code**

<https://github.com/Vilinz/octconv>

 

 

### References

<https://github.com/d-li14/octconv.pytorch>