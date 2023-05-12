# LeNet5

Train model,use [train2.txt](https://github.com/Robert0831/NNClassfication/blob/main/train2.txt) data ,it is shuffled training set ,and you need to put the data in the folder [images](https://github.com/Robert0831/NNClassfication/tree/main/image)

Trining
----------------------------------------------------------------------------
To train pytorch:[lenet_tor.py](https://github.com/Robert0831/LeNet5/blob/main/lenet_tor.py)

To train tensorflow:[lenet_tf.py](https://github.com/Robert0831/LeNet5/blob/main/lenet_tf.py)

To train tensorflow-static:[lenet_sta.py](https://github.com/Robert0831/LeNet5/blob/main/lenet_sta.py)

To train handcrafted:[Lenet.py](https://github.com/Robert0831/LeNet5/blob/main/Lenet.py)

Testing
----------------------------------------------------------------------------

To test model execution [test.py](https://github.com/Robert0831/LeNet5/blob/main/test.py)
and [test_handcrafted.py](https://github.com/Robert0831/LeNet5/blob/main/test_handcrafted.py) (for handcraft-model)


Pretrain model 
----------------------------------------------------------------------------

LeNet5-handcrafted:[Lanet_weights.pkl](https://github.com/Robert0831/LeNet5/blob/main/Lanet_weights.pkl)

LeNet5-pytorch:[tor_model_49.pth](https://github.com/Robert0831/LeNet5/blob/main/tor_model_49.pth)

LeNet5-tensorflow:[tf_model_49.keras](https://github.com/Robert0831/LeNet5/blob/main/tf_model_49.keras)

LeNet5-tensorflow_static:[tf_model_static.keras](https://github.com/Robert0831/LeNet5/blob/main/tf_model_static.keras)

pytorh_Post Training Dynamic Quantization:[quantized_model.pth](https://github.com/Robert0831/LeNet5/blob/main/quantized_model.pth)

pytorh_Post Pruning:[quantized_model_1.pth](https://github.com/Robert0831/LeNet5/blob/main/quantized_model_1.pth)

Model compression
----------------------------------------------------------------------------
Use [tor_qat.py](https://github.com/Robert0831/LeNet5/blob/main/tor_qat.py) to do Post Training Dynamic Quantization or Pruning

Reference
----------------------------------------------------------------------------
Reference code: https://github.com/toxtli/lenet-5-mnist-from-scratch-numpy
