**AI image classification using a Deep Convolulational Neural Net**
**By Reese Johnson**

**neccesary requirements for local deployment:**
- pip install tensorflow
- pip install keras
- pip install opencv-python
- pip install scikit-learn

**required data set**
- please download from https://www.kaggle.com/datasets/awsaf49/artifact-dataset


**Description**
Welcome to an AI image detection project. I began by training a logistic regression (logistic regression.py) using all images in the kaggle dataset - which got about 57% accuracy. I then realized a convolutional Neural Network is much better for processing images and recognizing patterns in the pixels that correlate with the meaning of the image. I then ran my CNN (deep_cnn.py) on all images in the bank - batched randomly with 1,000 images per batch (500 real, 500 fake) and trained it over 50 batches (50,000 images). The CNN has about 4 million parameters. The average test accuracy across the batches was 63.4% which was not bad, and already close to my goal. The issue I realized was this dataset had too vast of a variety of images for the CNN to properly train itself and understand clear correlations. For example, some folders contain AI generated images of artwork - which is very confusing for a Neural Network to decypher. As a human looking at such a broad scope of image categories I became confused. How was I to know if a painting was AI generated or just a real painting? Similarly with landscapes, animals, products, etc. Thus I attempted to narrow the scope of my project to train the CNN on a specific type of image: human faces. There were  8 different folders of AI generated faces from different AI architectures to create them. Some architectures are much better then others - ie Generative Adversarial Networks have very convincing images. I encourage whoever is reading this to checkout styleGANs AI generated faces - nearly indistinguishable from real people at the naked eye. Below you can see my CNNs average performance against various different types of AI architectures. I was very happy to see my CNN exceed 90% accuracy in some cases. I then attempted to mix up all AI architectures into one testing batch to see how my CNN can generalize - this proved to be a fairly difficult task. I believe I could have achieved an even higher accuracy if I had access to a better GPU and more memory.



**example output from trial 1: (tested on 50,000 images)**
(tensorflow) Reeses-MacBook-Pro:FinalProject reesejohnson$ python3 deep_cnn.py

computer vision cv2 version: 4.5.5
770596 real images processed for training
1379144 fake images processed for training
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 197, 197, 32)      1568      
                                                                 
 flatten (Flatten)           (None, 1241888)           0         
                                                                 
 dense (Dense)               (None, 32)                39740448  
                                                                 
 dense_1 (Dense)             (None, 2)                 66        
                                                                 
=================================================================
Total params: 39742082 (151.60 MB)
Trainable params: 39742082 (151.60 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
5/5 [==============================] - 17s 3s/step - loss: 24.3909 - acc: 0.5040 - val_loss: 10.9126 - val_acc: 0.4990
5/5 [==============================] - 15s 3s/step - loss: 5.0775 - acc: 0.5260 - val_loss: 4.1550 - val_acc: 0.5140
5/5 [==============================] - 17s 3s/step - loss: 2.4219 - acc: 0.5330 - val_loss: 1.0044 - val_acc: 0.5830
5/5 [==============================] - 14s 3s/step - loss: 1.2121 - acc: 0.5740 - val_loss: 0.8085 - val_acc: 0.6060
5/5 [==============================] - 28s 6s/step - loss: 0.8719 - acc: 0.5500 - val_loss: 0.7067 - val_acc: 0.6380
5/5 [==============================] - 17s 3s/step - loss: 0.7584 - acc: 0.5810 - val_loss: 0.7252 - val_acc: 0.5850
5/5 [==============================] - 16s 3s/step - loss: 0.7142 - acc: 0.6050 - val_loss: 0.6789 - val_acc: 0.6270
5/5 [==============================] - 17s 3s/step - loss: 0.6744 - acc: 0.5970 - val_loss: 0.6832 - val_acc: 0.6360
5/5 [==============================] - 18s 3s/step - loss: 0.6936 - acc: 0.6020 - val_loss: 0.6689 - val_acc: 0.6160
5/5 [==============================] - 17s 3s/step - loss: 0.6686 - acc: 0.6330 - val_loss: 0.6799 - val_acc: 0.6020
5/5 [==============================] - 16s 3s/step - loss: 0.6648 - acc: 0.6190 - val_loss: 0.6655 - val_acc: 0.6370
5/5 [==============================] - 16s 3s/step - loss: 0.6605 - acc: 0.6420 - val_loss: 0.6533 - val_acc: 0.6330
5/5 [==============================] - 15s 3s/step - loss: 0.6584 - acc: 0.6140 - val_loss: 0.6564 - val_acc: 0.6330
5/5 [==============================] - 17s 3s/step - loss: 0.6654 - acc: 0.5910 - val_loss: 0.6719 - val_acc: 0.5900
5/5 [==============================] - 16s 3s/step - loss: 0.6491 - acc: 0.6470 - val_loss: 0.6648 - val_acc: 0.6020
5/5 [==============================] - 16s 3s/step - loss: 0.6396 - acc: 0.6470 - val_loss: 0.6694 - val_acc: 0.6050
5/5 [==============================] - 17s 3s/step - loss: 0.6460 - acc: 0.6400 - val_loss: 0.6547 - val_acc: 0.6410
5/5 [==============================] - 17s 3s/step - loss: 0.6604 - acc: 0.6100 - val_loss: 0.6335 - val_acc: 0.6530
5/5 [==============================] - 16s 3s/step - loss: 0.6687 - acc: 0.6350 - val_loss: 0.6651 - val_acc: 0.6120
5/5 [==============================] - 17s 3s/step - loss: 0.6353 - acc: 0.6450 - val_loss: 0.6657 - val_acc: 0.5970
5/5 [==============================] - 18s 4s/step - loss: 0.6490 - acc: 0.6210 - val_loss: 0.6285 - val_acc: 0.6340
5/5 [==============================] - 17s 3s/step - loss: 0.6366 - acc: 0.6420 - val_loss: 0.6580 - val_acc: 0.6020
5/5 [==============================] - 17s 3s/step - loss: 0.6420 - acc: 0.6550 - val_loss: 0.6544 - val_acc: 0.6060
5/5 [==============================] - 16s 3s/step - loss: 0.6453 - acc: 0.6470 - val_loss: 0.6718 - val_acc: 0.6000
5/5 [==============================] - 16s 3s/step - loss: 0.6615 - acc: 0.6180 - val_loss: 0.6516 - val_acc: 0.6390
5/5 [==============================] - 17s 3s/step - loss: 0.6470 - acc: 0.6260 - val_loss: 0.6429 - val_acc: 0.6310
5/5 [==============================] - 17s 3s/step - loss: 0.6407 - acc: 0.6390 - val_loss: 0.6575 - val_acc: 0.6110
5/5 [==============================] - 16s 3s/step - loss: 0.6307 - acc: 0.6480 - val_loss: 0.6538 - val_acc: 0.6450
5/5 [==============================] - 17s 3s/step - loss: 0.6519 - acc: 0.6250 - val_loss: 0.6614 - val_acc: 0.6260
5/5 [==============================] - 19s 3s/step - loss: 0.6418 - acc: 0.6400 - val_loss: 0.6553 - val_acc: 0.6260
5/5 [==============================] - 19s 3s/step - loss: 0.6562 - acc: 0.6070 - val_loss: 0.6579 - val_acc: 0.6290
5/5 [==============================] - 16s 3s/step - loss: 0.6280 - acc: 0.6530 - val_loss: 0.6490 - val_acc: 0.6290
5/5 [==============================] - 16s 3s/step - loss: 0.6439 - acc: 0.6300 - val_loss: 0.6736 - val_acc: 0.5990
5/5 [==============================] - 19s 3s/step - loss: 0.6421 - acc: 0.6370 - val_loss: 0.6359 - val_acc: 0.6620
5/5 [==============================] - 18s 3s/step - loss: 0.6414 - acc: 0.6370 - val_loss: 0.6601 - val_acc: 0.6160
5/5 [==============================] - 17s 3s/step - loss: 0.6428 - acc: 0.6380 - val_loss: 0.6467 - val_acc: 0.6290
5/5 [==============================] - 17s 3s/step - loss: 0.6339 - acc: 0.6450 - val_loss: 0.6371 - val_acc: 0.6440
5/5 [==============================] - 18s 3s/step - loss: 0.6438 - acc: 0.6480 - val_loss: 0.6496 - val_acc: 0.6300
5/5 [==============================] - 16s 3s/step - loss: 0.6468 - acc: 0.6400 - val_loss: 0.6524 - val_acc: 0.6270
5/5 [==============================] - 17s 3s/step - loss: 0.6535 - acc: 0.6250 - val_loss: 0.6308 - val_acc: 0.6690
5/5 [==============================] - 19s 3s/step - loss: 0.6325 - acc: 0.6520 - val_loss: 0.6466 - val_acc: 0.6230
5/5 [==============================] - 17s 3s/step - loss: 0.6513 - acc: 0.6210 - val_loss: 0.6455 - val_acc: 0.6350
5/5 [==============================] - 17s 3s/step - loss: 0.6486 - acc: 0.6210 - val_loss: 0.6375 - val_acc: 0.6330
5/5 [==============================] - 16s 3s/step - loss: 0.6488 - acc: 0.6310 - val_loss: 0.6378 - val_acc: 0.6380
5/5 [==============================] - 17s 3s/step - loss: 0.6161 - acc: 0.6380 - val_loss: 0.6428 - val_acc: 0.6350
5/5 [==============================] - 17s 3s/step - loss: 0.6444 - acc: 0.6200 - val_loss: 0.6223 - val_acc: 0.6570
5/5 [==============================] - 17s 3s/step - loss: 0.6468 - acc: 0.6260 - val_loss: 0.6417 - val_acc: 0.6620
5/5 [==============================] - 22s 4s/step - loss: 0.6441 - acc: 0.6270 - val_loss: 0.6227 - val_acc: 0.6570
5/5 [==============================] - 18s 4s/step - loss: 0.6618 - acc: 0.6090 - val_loss: 0.6636 - val_acc: 0.6100
5/5 [==============================] - 17s 3s/step - loss: 0.6525 - acc: 0.6210 - val_loss: 0.6424 - val_acc: 0.6280
32/32 [==============================] - 3s 85ms/step - loss: 0.6548 - acc: 0.6340
Final accuracy on test data:  0.6340000033378601
Final loss on test data:  0.6548075079917908


**example output 2: real images of faces vs syntehtically generated (low quality) faces from face synthetics**
computer vision cv2 version: 4.5.5
100000 real images processed for training
10000 fake images processed for training
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 197, 197, 32)      1568      
                                                                 
 flatten (Flatten)           (None, 1241888)           0         
                                                                 
 dense (Dense)               (None, 32)                39740448  
                                                                 
 dense_1 (Dense)             (None, 2)                 66        
                                                                 
=================================================================
Total params: 39742082 (151.60 MB)
Trainable params: 39742082 (151.60 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
5/5 [==============================] - 22s 3s/step - loss: 38.8326 - acc: 0.5590 - val_loss: 52.4746 - val_acc: 0.5000
5/5 [==============================] - 17s 3s/step - loss: 25.6032 - acc: 0.5810 - val_loss: 14.1673 - val_acc: 0.5280
5/5 [==============================] - 15s 3s/step - loss: 8.3080 - acc: 0.5990 - val_loss: 1.2803 - val_acc: 0.7910
5/5 [==============================] - 14s 3s/step - loss: 1.9617 - acc: 0.6870 - val_loss: 1.4043 - val_acc: 0.6280
5/5 [==============================] - 16s 3s/step - loss: 1.0872 - acc: 0.6630 - val_loss: 0.7112 - val_acc: 0.7270
5/5 [==============================] - 16s 3s/step - loss: 0.6035 - acc: 0.7690 - val_loss: 0.5893 - val_acc: 0.8130
5/5 [==============================] - 15s 3s/step - loss: 0.5500 - acc: 0.8230 - val_loss: 0.5037 - val_acc: 0.8350
5/5 [==============================] - 17s 3s/step - loss: 0.5166 - acc: 0.8160 - val_loss: 0.4483 - val_acc: 0.8420
5/5 [==============================] - 16s 3s/step - loss: 0.3957 - acc: 0.8720 - val_loss: 0.2943 - val_acc: 0.8960
5/5 [==============================] - 17s 3s/step - loss: 0.3129 - acc: 0.8860 - val_loss: 0.2471 - val_acc: 0.9080
5/5 [==============================] - 14s 3s/step - loss: 0.2858 - acc: 0.8930 - val_loss: 0.3390 - val_acc: 0.8850
5/5 [==============================] - 16s 3s/step - loss: 0.3063 - acc: 0.8850 - val_loss: 0.3360 - val_acc: 0.8710
5/5 [==============================] - 16s 3s/step - loss: 0.2256 - acc: 0.9110 - val_loss: 0.2574 - val_acc: 0.8990
5/5 [==============================] - 16s 3s/step - loss: 0.2425 - acc: 0.9140 - val_loss: 0.2231 - val_acc: 0.9140
5/5 [==============================] - 15s 3s/step - loss: 0.2161 - acc: 0.9100 - val_loss: 0.1944 - val_acc: 0.9340
5/5 [==============================] - 15s 3s/step - loss: 0.2112 - acc: 0.9280 - val_loss: 0.2056 - val_acc: 0.9230
5/5 [==============================] - 15s 3s/step - loss: 0.2183 - acc: 0.9260 - val_loss: 0.1625 - val_acc: 0.9460
5/5 [==============================] - 16s 3s/step - loss: 0.2013 - acc: 0.9180 - val_loss: 0.1926 - val_acc: 0.9220
5/5 [==============================] - 14s 3s/step - loss: 0.1762 - acc: 0.9430 - val_loss: 0.1436 - val_acc: 0.9400
5/5 [==============================] - 14s 3s/step - loss: 0.1297 - acc: 0.9460 - val_loss: 0.1242 - val_acc: 0.9560
5/5 [==============================] - 18s 3s/step - loss: 0.1069 - acc: 0.9580 - val_loss: 0.1366 - val_acc: 0.9470
5/5 [==============================] - 14s 3s/step - loss: 0.1360 - acc: 0.9500 - val_loss: 0.1581 - val_acc: 0.9390
5/5 [==============================] - 16s 3s/step - loss: 0.1455 - acc: 0.9480 - val_loss: 0.1386 - val_acc: 0.9520
5/5 [==============================] - 15s 3s/step - loss: 0.1321 - acc: 0.9500 - val_loss: 0.1470 - val_acc: 0.9450
5/5 [==============================] - 16s 3s/step - loss: 0.1409 - acc: 0.9510 - val_loss: 0.1379 - val_acc: 0.9460
5/5 [==============================] - 15s 3s/step - loss: 0.1044 - acc: 0.9630 - val_loss: 0.1322 - val_acc: 0.9500
5/5 [==============================] - 16s 3s/step - loss: 0.0783 - acc: 0.9770 - val_loss: 0.1418 - val_acc: 0.9480
5/5 [==============================] - 15s 3s/step - loss: 0.1101 - acc: 0.9600 - val_loss: 0.1053 - val_acc: 0.9620
5/5 [==============================] - 15s 3s/step - loss: 0.1010 - acc: 0.9640 - val_loss: 0.1157 - val_acc: 0.9500
5/5 [==============================] - 15s 3s/step - loss: 0.1113 - acc: 0.9570 - val_loss: 0.1654 - val_acc: 0.9510
5/5 [==============================] - 16s 3s/step - loss: 0.0803 - acc: 0.9690 - val_loss: 0.1126 - val_acc: 0.9630
5/5 [==============================] - 13s 3s/step - loss: 0.0823 - acc: 0.9720 - val_loss: 0.1534 - val_acc: 0.9500
5/5 [==============================] - 15s 3s/step - loss: 0.0743 - acc: 0.9800 - val_loss: 0.1310 - val_acc: 0.9520
5/5 [==============================] - 16s 3s/step - loss: 0.0848 - acc: 0.9630 - val_loss: 0.1065 - val_acc: 0.9680
5/5 [==============================] - 16s 3s/step - loss: 0.0894 - acc: 0.9620 - val_loss: 0.1380 - val_acc: 0.9410
5/5 [==============================] - 16s 3s/step - loss: 0.0881 - acc: 0.9660 - val_loss: 0.1330 - val_acc: 0.9560
5/5 [==============================] - 14s 3s/step - loss: 0.0921 - acc: 0.9710 - val_loss: 0.1114 - val_acc: 0.9530
5/5 [==============================] - 17s 3s/step - loss: 0.0731 - acc: 0.9760 - val_loss: 0.2608 - val_acc: 0.9060
5/5 [==============================] - 16s 3s/step - loss: 0.1102 - acc: 0.9630 - val_loss: 0.1274 - val_acc: 0.9490
5/5 [==============================] - 16s 3s/step - loss: 0.0818 - acc: 0.9660 - val_loss: 0.1141 - val_acc: 0.9550
5/5 [==============================] - 18s 3s/step - loss: 0.0677 - acc: 0.9780 - val_loss: 0.0906 - val_acc: 0.9660
5/5 [==============================] - 15s 3s/step - loss: 0.0643 - acc: 0.9800 - val_loss: 0.1101 - val_acc: 0.9620
5/5 [==============================] - 16s 3s/step - loss: 0.0609 - acc: 0.9760 - val_loss: 0.0792 - val_acc: 0.9680
5/5 [==============================] - 17s 3s/step - loss: 0.0497 - acc: 0.9840 - val_loss: 0.1126 - val_acc: 0.9600
5/5 [==============================] - 14s 3s/step - loss: 0.0533 - acc: 0.9850 - val_loss: 0.1162 - val_acc: 0.9570
5/5 [==============================] - 16s 3s/step - loss: 0.0473 - acc: 0.9840 - val_loss: 0.0801 - val_acc: 0.9690
5/5 [==============================] - 15s 3s/step - loss: 0.0580 - acc: 0.9760 - val_loss: 0.1209 - val_acc: 0.9530
5/5 [==============================] - 15s 3s/step - loss: 0.0534 - acc: 0.9780 - val_loss: 0.0976 - val_acc: 0.9650
5/5 [==============================] - 15s 3s/step - loss: 0.0414 - acc: 0.9880 - val_loss: 0.1191 - val_acc: 0.9600
5/5 [==============================] - 16s 3s/step - loss: 0.0495 - acc: 0.9850 - val_loss: 0.1559 - val_acc: 0.9440
32/32 [==============================] - 3s 85ms/step - loss: 0.1434 - acc: 0.9620
Final accuracy on test data:  0.9620000123977661
Final loss on test data:  0.1434118002653122


**example of trial 3: real faces vs palette faces:**
computer vision cv2 version: 4.5.5
100000 real images processed for training
6000 fake images processed for training
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 197, 197, 32)      1568      
                                                                 
 flatten (Flatten)           (None, 1241888)           0         
                                                                 
 dense (Dense)               (None, 32)                39740448  
                                                                 
 dense_1 (Dense)             (None, 2)                 66        
                                                                 
=================================================================
Total params: 39742082 (151.60 MB)
Trainable params: 39742082 (151.60 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
5/5 [==============================] - 17s 3s/step - loss: 57.4515 - acc: 0.4760 - val_loss: 50.8469 - val_acc: 0.5000
5/5 [==============================] - 15s 3s/step - loss: 44.8985 - acc: 0.5000 - val_loss: 8.5342 - val_acc: 0.5000
5/5 [==============================] - 18s 4s/step - loss: 15.5150 - acc: 0.5000 - val_loss: 7.5891 - val_acc: 0.5000
5/5 [==============================] - 17s 3s/step - loss: 3.9430 - acc: 0.5040 - val_loss: 4.1228 - val_acc: 0.5010
5/5 [==============================] - 18s 4s/step - loss: 2.7139 - acc: 0.5020 - val_loss: 0.8016 - val_acc: 0.5080
5/5 [==============================] - 17s 3s/step - loss: 0.7061 - acc: 0.5550 - val_loss: 0.7014 - val_acc: 0.5010
5/5 [==============================] - 15s 3s/step - loss: 0.6779 - acc: 0.4990 - val_loss: 0.6975 - val_acc: 0.4990
5/5 [==============================] - 16s 3s/step - loss: 0.6800 - acc: 0.5010 - val_loss: 0.6908 - val_acc: 0.5010
5/5 [==============================] - 16s 3s/step - loss: 0.6788 - acc: 0.5060 - val_loss: 0.6823 - val_acc: 0.5110
5/5 [==============================] - 17s 3s/step - loss: 0.6718 - acc: 0.5220 - val_loss: 0.6753 - val_acc: 0.5120
5/5 [==============================] - 17s 3s/step - loss: 0.6655 - acc: 0.5530 - val_loss: 0.6714 - val_acc: 0.5710
5/5 [==============================] - 15s 3s/step - loss: 0.6507 - acc: 0.6200 - val_loss: 0.6828 - val_acc: 0.5850
5/5 [==============================] - 16s 3s/step - loss: 0.6464 - acc: 0.6200 - val_loss: 0.6648 - val_acc: 0.5730
5/5 [==============================] - 17s 3s/step - loss: 0.6275 - acc: 0.6790 - val_loss: 0.6588 - val_acc: 0.6280
5/5 [==============================] - 16s 3s/step - loss: 0.6052 - acc: 0.6940 - val_loss: 0.6502 - val_acc: 0.6170
5/5 [==============================] - 15s 3s/step - loss: 0.5722 - acc: 0.7440 - val_loss: 0.6552 - val_acc: 0.6250
5/5 [==============================] - 18s 3s/step - loss: 0.7370 - acc: 0.6420 - val_loss: 0.7090 - val_acc: 0.5320
5/5 [==============================] - 16s 3s/step - loss: 0.5690 - acc: 0.7250 - val_loss: 0.6698 - val_acc: 0.6100
5/5 [==============================] - 17s 3s/step - loss: 0.5948 - acc: 0.6960 - val_loss: 0.6592 - val_acc: 0.6160
5/5 [==============================] - 16s 3s/step - loss: 0.5934 - acc: 0.6930 - val_loss: 0.6832 - val_acc: 0.5790
5/5 [==============================] - 17s 3s/step - loss: 0.5596 - acc: 0.7470 - val_loss: 0.6642 - val_acc: 0.5930
5/5 [==============================] - 15s 3s/step - loss: 0.5512 - acc: 0.7430 - val_loss: 0.6740 - val_acc: 0.6030
5/5 [==============================] - 15s 3s/step - loss: 0.5295 - acc: 0.7440 - val_loss: 0.6593 - val_acc: 0.6150
5/5 [==============================] - 16s 3s/step - loss: 0.5136 - acc: 0.7510 - val_loss: 0.7079 - val_acc: 0.6050
5/5 [==============================] - 17s 3s/step - loss: 0.5217 - acc: 0.7450 - val_loss: 0.6825 - val_acc: 0.6180
5/5 [==============================] - 15s 3s/step - loss: 0.5252 - acc: 0.7590 - val_loss: 0.6647 - val_acc: 0.6080
5/5 [==============================] - 16s 3s/step - loss: 0.4988 - acc: 0.7820 - val_loss: 0.7369 - val_acc: 0.5940
5/5 [==============================] - 15s 3s/step - loss: 0.4883 - acc: 0.7760 - val_loss: 0.6677 - val_acc: 0.6420
5/5 [==============================] - 14s 3s/step - loss: 0.4680 - acc: 0.7890 - val_loss: 0.6548 - val_acc: 0.6200
5/5 [==============================] - 16s 3s/step - loss: 0.4664 - acc: 0.8030 - val_loss: 0.6550 - val_acc: 0.6570
5/5 [==============================] - 14s 3s/step - loss: 0.4302 - acc: 0.8220 - val_loss: 0.7186 - val_acc: 0.6160
5/5 [==============================] - 15s 3s/step - loss: 0.4110 - acc: 0.8200 - val_loss: 0.6635 - val_acc: 0.6500
5/5 [==============================] - 15s 3s/step - loss: 0.3950 - acc: 0.8540 - val_loss: 0.6379 - val_acc: 0.6660
5/5 [==============================] - 15s 3s/step - loss: 0.3717 - acc: 0.8530 - val_loss: 0.8019 - val_acc: 0.6190
5/5 [==============================] - 16s 3s/step - loss: 0.3676 - acc: 0.8500 - val_loss: 0.7850 - val_acc: 0.6130
5/5 [==============================] - 15s 3s/step - loss: 0.3702 - acc: 0.8390 - val_loss: 0.7935 - val_acc: 0.5990
5/5 [==============================] - 16s 3s/step - loss: 0.3736 - acc: 0.8540 - val_loss: 0.7203 - val_acc: 0.6430
5/5 [==============================] - 16s 3s/step - loss: 0.3274 - acc: 0.8740 - val_loss: 0.7634 - val_acc: 0.6320
5/5 [==============================] - 16s 3s/step - loss: 0.2985 - acc: 0.8910 - val_loss: 0.7329 - val_acc: 0.6470
5/5 [==============================] - 16s 3s/step - loss: 0.3136 - acc: 0.8780 - val_loss: 0.7027 - val_acc: 0.6480
5/5 [==============================] - 15s 3s/step - loss: 0.2913 - acc: 0.9000 - val_loss: 0.7389 - val_acc: 0.6610
5/5 [==============================] - 15s 3s/step - loss: 0.3554 - acc: 0.8400 - val_loss: 0.6895 - val_acc: 0.6560
5/5 [==============================] - 17s 3s/step - loss: 0.3181 - acc: 0.8920 - val_loss: 0.7611 - val_acc: 0.6140
5/5 [==============================] - 15s 3s/step - loss: 0.3121 - acc: 0.9010 - val_loss: 0.7405 - val_acc: 0.6440
5/5 [==============================] - 16s 3s/step - loss: 0.2806 - acc: 0.9060 - val_loss: 0.9183 - val_acc: 0.6160
5/5 [==============================] - 16s 3s/step - loss: 0.2771 - acc: 0.8910 - val_loss: 0.8187 - val_acc: 0.6600
5/5 [==============================] - 15s 3s/step - loss: 0.2543 - acc: 0.9090 - val_loss: 0.7558 - val_acc: 0.6590
5/5 [==============================] - 16s 3s/step - loss: 0.2380 - acc: 0.9190 - val_loss: 0.8241 - val_acc: 0.6760
5/5 [==============================] - 15s 3s/step - loss: 0.2497 - acc: 0.9050 - val_loss: 0.8397 - val_acc: 0.6310
5/5 [==============================] - 17s 3s/step - loss: 0.2269 - acc: 0.9270 - val_loss: 0.9223 - val_acc: 0.6350
32/32 [==============================] - 3s 78ms/step - loss: 0.7954 - acc: 0.6710
Final accuracy on test data:  0.6710000038146973
Final loss on test data:  0.7954234480857849


**example 4 trial: real faces vs sfhq dataset:**
computer vision cv2 version: 4.5.5
100000 real images processed for training
10000 fake images processed for training
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 197, 197, 32)      1568      
                                                                 
 flatten (Flatten)           (None, 1241888)           0         
                                                                 
 dense (Dense)               (None, 32)                39740448  
                                                                 
 dense_1 (Dense)             (None, 2)                 66        
                                                                 
=================================================================
Total params: 39742082 (151.60 MB)
Trainable params: 39742082 (151.60 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
5/5 [==============================] - 25s 4s/step - loss: 22.7226 - acc: 0.5060 - val_loss: 13.0152 - val_acc: 0.5010
5/5 [==============================] - 26s 5s/step - loss: 9.0198 - acc: 0.5080 - val_loss: 4.0034 - val_acc: 0.5220
5/5 [==============================] - 30s 5s/step - loss: 3.0004 - acc: 0.5290 - val_loss: 2.3409 - val_acc: 0.5080
5/5 [==============================] - 29s 5s/step - loss: 1.0505 - acc: 0.6020 - val_loss: 0.5710 - val_acc: 0.6730
5/5 [==============================] - 22s 4s/step - loss: 0.6024 - acc: 0.6650 - val_loss: 0.6204 - val_acc: 0.6920
5/5 [==============================] - 25s 4s/step - loss: 0.5998 - acc: 0.6760 - val_loss: 0.6269 - val_acc: 0.6120
5/5 [==============================] - 23s 4s/step - loss: 0.6185 - acc: 0.6860 - val_loss: 0.5808 - val_acc: 0.7130
5/5 [==============================] - 22s 4s/step - loss: 0.6096 - acc: 0.6960 - val_loss: 0.5763 - val_acc: 0.6850
5/5 [==============================] - 24s 4s/step - loss: 0.6023 - acc: 0.6420 - val_loss: 0.5524 - val_acc: 0.7160
5/5 [==============================] - 25s 4s/step - loss: 0.5502 - acc: 0.7000 - val_loss: 0.5334 - val_acc: 0.6660
5/5 [==============================] - 23s 5s/step - loss: 0.4822 - acc: 0.7610 - val_loss: 0.4798 - val_acc: 0.7640
5/5 [==============================] - 42s 9s/step - loss: 0.4996 - acc: 0.7640 - val_loss: 0.5667 - val_acc: 0.7530
5/5 [==============================] - 25s 4s/step - loss: 0.4635 - acc: 0.8130 - val_loss: 0.4487 - val_acc: 0.8240
5/5 [==============================] - 25s 4s/step - loss: 0.4482 - acc: 0.8220 - val_loss: 0.4362 - val_acc: 0.8380
5/5 [==============================] - 29s 6s/step - loss: 0.3986 - acc: 0.8440 - val_loss: 0.4354 - val_acc: 0.8420
5/5 [==============================] - 26s 4s/step - loss: 0.3819 - acc: 0.8640 - val_loss: 0.3468 - val_acc: 0.8710
5/5 [==============================] - 39s 6s/step - loss: 0.3470 - acc: 0.8580 - val_loss: 0.3387 - val_acc: 0.8670
5/5 [==============================] - 25s 5s/step - loss: 0.3440 - acc: 0.8760 - val_loss: 0.3422 - val_acc: 0.8610
5/5 [==============================] - 25s 5s/step - loss: 0.2657 - acc: 0.9030 - val_loss: 0.3224 - val_acc: 0.8870
5/5 [==============================] - 24s 4s/step - loss: 0.3026 - acc: 0.8820 - val_loss: 0.3016 - val_acc: 0.8790
5/5 [==============================] - 26s 5s/step - loss: 0.2553 - acc: 0.8950 - val_loss: 0.2832 - val_acc: 0.8880
5/5 [==============================] - 25s 5s/step - loss: 0.2432 - acc: 0.9100 - val_loss: 0.2824 - val_acc: 0.8890
5/5 [==============================] - 23s 4s/step - loss: 0.2550 - acc: 0.9000 - val_loss: 0.2989 - val_acc: 0.8800
5/5 [==============================] - 24s 4s/step - loss: 0.2302 - acc: 0.9140 - val_loss: 0.2589 - val_acc: 0.9050
5/5 [==============================] - 24s 4s/step - loss: 0.2128 - acc: 0.9320 - val_loss: 0.2612 - val_acc: 0.8970
5/5 [==============================] - 29s 5s/step - loss: 0.2006 - acc: 0.9290 - val_loss: 0.2702 - val_acc: 0.8920
5/5 [==============================] - 22s 4s/step - loss: 0.2147 - acc: 0.9250 - val_loss: 0.2368 - val_acc: 0.9040
5/5 [==============================] - 27s 6s/step - loss: 0.1812 - acc: 0.9300 - val_loss: 0.2762 - val_acc: 0.8960
5/5 [==============================] - 20s 4s/step - loss: 0.2061 - acc: 0.9170 - val_loss: 0.2975 - val_acc: 0.8790
5/5 [==============================] - 25s 4s/step - loss: 0.1880 - acc: 0.9320 - val_loss: 0.2399 - val_acc: 0.8970
5/5 [==============================] - 28s 5s/step - loss: 0.1386 - acc: 0.9520 - val_loss: 0.2327 - val_acc: 0.9010
5/5 [==============================] - 21s 4s/step - loss: 0.1879 - acc: 0.9390 - val_loss: 0.2532 - val_acc: 0.9130
5/5 [==============================] - 32s 7s/step - loss: 0.1796 - acc: 0.9340 - val_loss: 0.2642 - val_acc: 0.9000
5/5 [==============================] - 25s 4s/step - loss: 0.1661 - acc: 0.9390 - val_loss: 0.2497 - val_acc: 0.8960
5/5 [==============================] - 25s 4s/step - loss: 0.1797 - acc: 0.9420 - val_loss: 0.2597 - val_acc: 0.8950
5/5 [==============================] - 22s 4s/step - loss: 0.1538 - acc: 0.9490 - val_loss: 0.2922 - val_acc: 0.8950
5/5 [==============================] - 20s 4s/step - loss: 0.1568 - acc: 0.9470 - val_loss: 0.2399 - val_acc: 0.8950
5/5 [==============================] - 19s 4s/step - loss: 0.1406 - acc: 0.9570 - val_loss: 0.2509 - val_acc: 0.9070
5/5 [==============================] - 25s 4s/step - loss: 0.1589 - acc: 0.9480 - val_loss: 0.2586 - val_acc: 0.8980
5/5 [==============================] - 21s 4s/step - loss: 0.1249 - acc: 0.9620 - val_loss: 0.2323 - val_acc: 0.9120
5/5 [==============================] - 26s 5s/step - loss: 0.1338 - acc: 0.9550 - val_loss: 0.2123 - val_acc: 0.9230
5/5 [==============================] - 40s 8s/step - loss: 0.1165 - acc: 0.9570 - val_loss: 0.2282 - val_acc: 0.9140
5/5 [==============================] - 27s 5s/step - loss: 0.1141 - acc: 0.9580 - val_loss: 0.2326 - val_acc: 0.9130
5/5 [==============================] - 31s 6s/step - loss: 0.1095 - acc: 0.9620 - val_loss: 0.2153 - val_acc: 0.9100
5/5 [==============================] - 26s 5s/step - loss: 0.1383 - acc: 0.9520 - val_loss: 0.2583 - val_acc: 0.9010
5/5 [==============================] - 45s 7s/step - loss: 0.1348 - acc: 0.9580 - val_loss: 0.3434 - val_acc: 0.8600
5/5 [==============================] - 28s 5s/step - loss: 0.1242 - acc: 0.9540 - val_loss: 0.2121 - val_acc: 0.9140
5/5 [==============================] - 29s 5s/step - loss: 0.1520 - acc: 0.9500 - val_loss: 0.2264 - val_acc: 0.9120
5/5 [==============================] - 32s 5s/step - loss: 0.1328 - acc: 0.9560 - val_loss: 0.2884 - val_acc: 0.8960
5/5 [==============================] - 44s 8s/step - loss: 0.1124 - acc: 0.9620 - val_loss: 0.2937 - val_acc: 0.8980
32/32 [==============================] - 5s 127ms/step - loss: 0.2820 - acc: 0.8970
Final accuracy on test data:  0.8970000147819519
Final loss on test data:  0.28196245431900024


**example 5: real faces vs diffusion GAN model:**
(tensorflow) Reeses-MacBook-Pro:FinalProject reesejohnson$ python3 deep_cnn.py
2024-04-30 01:34:21.515678: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
computer vision cv2 version: 4.5.5
100000 real images processed for training
15507 fake images processed for training
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 197, 197, 32)      1568      
                                                                 
 flatten (Flatten)           (None, 1241888)           0         
                                                                 
 dense (Dense)               (None, 32)                39740448  
                                                                 
 dense_1 (Dense)             (None, 2)                 66        
                                                                 
=================================================================
Total params: 39742082 (151.60 MB)
Trainable params: 39742082 (151.60 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
5/5 [==============================] - 17s 3s/step - loss: 30.0224 - acc: 0.5100 - val_loss: 6.8555 - val_acc: 0.5400
5/5 [==============================] - 18s 3s/step - loss: 7.3124 - acc: 0.5170 - val_loss: 5.2882 - val_acc: 0.5090
5/5 [==============================] - 19s 3s/step - loss: 3.0344 - acc: 0.5750 - val_loss: 2.7946 - val_acc: 0.5310
5/5 [==============================] - 18s 3s/step - loss: 1.6127 - acc: 0.6340 - val_loss: 1.2370 - val_acc: 0.6620
5/5 [==============================] - 17s 3s/step - loss: 1.1193 - acc: 0.6680 - val_loss: 0.7170 - val_acc: 0.6880
5/5 [==============================] - 16s 3s/step - loss: 0.8567 - acc: 0.6210 - val_loss: 0.8362 - val_acc: 0.6120
5/5 [==============================] - 17s 3s/step - loss: 0.6612 - acc: 0.6810 - val_loss: 0.6465 - val_acc: 0.6900
5/5 [==============================] - 16s 3s/step - loss: 0.7112 - acc: 0.6490 - val_loss: 0.6799 - val_acc: 0.6870
5/5 [==============================] - 17s 3s/step - loss: 0.6629 - acc: 0.6830 - val_loss: 0.5991 - val_acc: 0.7120
5/5 [==============================] - 17s 3s/step - loss: 0.5654 - acc: 0.7610 - val_loss: 0.5452 - val_acc: 0.7570
5/5 [==============================] - 15s 3s/step - loss: 0.5570 - acc: 0.7400 - val_loss: 0.5565 - val_acc: 0.7530
5/5 [==============================] - 15s 3s/step - loss: 0.5261 - acc: 0.7550 - val_loss: 0.5024 - val_acc: 0.7720
5/5 [==============================] - 17s 3s/step - loss: 0.4950 - acc: 0.7680 - val_loss: 0.5094 - val_acc: 0.7340
5/5 [==============================] - 16s 3s/step - loss: 0.5236 - acc: 0.7540 - val_loss: 0.4803 - val_acc: 0.7940
5/5 [==============================] - 15s 3s/step - loss: 0.4880 - acc: 0.7860 - val_loss: 0.4579 - val_acc: 0.8080
5/5 [==============================] - 17s 3s/step - loss: 0.4492 - acc: 0.7930 - val_loss: 0.4504 - val_acc: 0.8100
5/5 [==============================] - 18s 4s/step - loss: 0.4664 - acc: 0.7810 - val_loss: 0.4810 - val_acc: 0.7730
5/5 [==============================] - 16s 3s/step - loss: 0.4850 - acc: 0.7910 - val_loss: 0.4853 - val_acc: 0.7820
5/5 [==============================] - 15s 3s/step - loss: 0.4787 - acc: 0.7820 - val_loss: 0.4777 - val_acc: 0.7750
5/5 [==============================] - 16s 3s/step - loss: 0.4942 - acc: 0.7750 - val_loss: 0.4477 - val_acc: 0.7970
5/5 [==============================] - 18s 3s/step - loss: 0.4414 - acc: 0.7880 - val_loss: 0.4629 - val_acc: 0.7840
5/5 [==============================] - 16s 3s/step - loss: 0.4702 - acc: 0.7700 - val_loss: 0.4586 - val_acc: 0.8000
5/5 [==============================] - 16s 3s/step - loss: 0.4574 - acc: 0.7910 - val_loss: 0.4962 - val_acc: 0.7760
5/5 [==============================] - 18s 4s/step - loss: 0.4885 - acc: 0.7630 - val_loss: 0.5255 - val_acc: 0.7530
5/5 [==============================] - 17s 3s/step - loss: 0.4740 - acc: 0.7780 - val_loss: 0.4767 - val_acc: 0.7540
5/5 [==============================] - 16s 3s/step - loss: 0.4620 - acc: 0.7790 - val_loss: 0.5119 - val_acc: 0.7590
5/5 [==============================] - 15s 3s/step - loss: 0.5403 - acc: 0.7110 - val_loss: 0.6138 - val_acc: 0.6960
5/5 [==============================] - 19s 4s/step - loss: 0.4681 - acc: 0.7790 - val_loss: 0.4584 - val_acc: 0.8100
5/5 [==============================] - 16s 3s/step - loss: 0.4625 - acc: 0.7830 - val_loss: 0.4741 - val_acc: 0.7820
5/5 [==============================] - 17s 3s/step - loss: 0.4304 - acc: 0.8110 - val_loss: 0.4969 - val_acc: 0.7610
5/5 [==============================] - 15s 3s/step - loss: 0.4726 - acc: 0.7850 - val_loss: 0.5013 - val_acc: 0.7730
5/5 [==============================] - 15s 3s/step - loss: 0.4539 - acc: 0.8010 - val_loss: 0.4789 - val_acc: 0.7910
5/5 [==============================] - 20s 4s/step - loss: 0.4503 - acc: 0.7980 - val_loss: 0.4397 - val_acc: 0.7950
5/5 [==============================] - 16s 3s/step - loss: 0.4292 - acc: 0.8010 - val_loss: 0.4742 - val_acc: 0.7730
5/5 [==============================] - 17s 3s/step - loss: 0.4342 - acc: 0.7960 - val_loss: 0.4342 - val_acc: 0.8000
5/5 [==============================] - 16s 3s/step - loss: 0.4336 - acc: 0.8000 - val_loss: 0.4679 - val_acc: 0.7970
5/5 [==============================] - 15s 3s/step - loss: 0.4545 - acc: 0.7920 - val_loss: 0.4271 - val_acc: 0.8060
5/5 [==============================] - 16s 3s/step - loss: 0.4158 - acc: 0.8000 - val_loss: 0.4353 - val_acc: 0.8030
5/5 [==============================] - 16s 3s/step - loss: 0.4225 - acc: 0.8000 - val_loss: 0.4252 - val_acc: 0.8270
5/5 [==============================] - 17s 3s/step - loss: 0.4084 - acc: 0.8220 - val_loss: 0.4529 - val_acc: 0.7880
5/5 [==============================] - 15s 3s/step - loss: 0.4092 - acc: 0.8090 - val_loss: 0.4189 - val_acc: 0.8080
5/5 [==============================] - 16s 3s/step - loss: 0.4211 - acc: 0.7890 - val_loss: 0.4578 - val_acc: 0.7850
5/5 [==============================] - 15s 3s/step - loss: 0.4172 - acc: 0.8050 - val_loss: 0.4090 - val_acc: 0.8180
5/5 [==============================] - 20s 3s/step - loss: 0.3751 - acc: 0.8360 - val_loss: 0.4251 - val_acc: 0.8090
5/5 [==============================] - 16s 3s/step - loss: 0.4133 - acc: 0.8120 - val_loss: 0.4377 - val_acc: 0.8080
5/5 [==============================] - 17s 3s/step - loss: 0.3873 - acc: 0.8320 - val_loss: 0.4339 - val_acc: 0.8130
5/5 [==============================] - 16s 3s/step - loss: 0.3973 - acc: 0.8280 - val_loss: 0.4354 - val_acc: 0.7990
5/5 [==============================] - 15s 3s/step - loss: 0.3960 - acc: 0.8310 - val_loss: 0.4603 - val_acc: 0.8090
5/5 [==============================] - 14s 3s/step - loss: 0.3924 - acc: 0.8270 - val_loss: 0.4254 - val_acc: 0.8040
5/5 [==============================] - 17s 3s/step - loss: 0.3644 - acc: 0.8440 - val_loss: 0.4155 - val_acc: 0.8200
32/32 [==============================] - 3s 83ms/step - loss: 0.4417 - acc: 0.7980
Final accuracy on test data:  0.7979999780654907
Final loss on test data:  0.4417422115802765



**example 6: real faces vs denoising diffusion GAN:**
computer vision cv2 version: 4.5.5
100000 real images processed for training
10000 fake images processed for training
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 197, 197, 32)      1568      
                                                                 
 flatten (Flatten)           (None, 1241888)           0         
                                                                 
 dense (Dense)               (None, 32)                39740448  
                                                                 
 dense_1 (Dense)             (None, 2)                 66        
                                                                 
=================================================================
Total params: 39742082 (151.60 MB)
Trainable params: 39742082 (151.60 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
5/5 [==============================] - 17s 3s/step - loss: 40.7523 - acc: 0.5020 - val_loss: 22.5752 - val_acc: 0.5000
5/5 [==============================] - 17s 3s/step - loss: 11.2460 - acc: 0.5410 - val_loss: 3.6585 - val_acc: 0.5100
5/5 [==============================] - 17s 3s/step - loss: 6.3158 - acc: 0.5050 - val_loss: 3.6420 - val_acc: 0.5020
5/5 [==============================] - 18s 3s/step - loss: 3.2374 - acc: 0.5210 - val_loss: 2.4391 - val_acc: 0.5090
5/5 [==============================] - 20s 4s/step - loss: 2.2649 - acc: 0.5270 - val_loss: 1.6151 - val_acc: 0.5300
5/5 [==============================] - 19s 3s/step - loss: 1.4487 - acc: 0.5320 - val_loss: 0.6588 - val_acc: 0.6040
5/5 [==============================] - 18s 3s/step - loss: 0.7347 - acc: 0.5570 - val_loss: 0.6929 - val_acc: 0.5640
5/5 [==============================] - 19s 3s/step - loss: 0.6680 - acc: 0.6000 - val_loss: 0.6794 - val_acc: 0.5350
5/5 [==============================] - 17s 3s/step - loss: 0.6625 - acc: 0.5800 - val_loss: 0.6655 - val_acc: 0.5420
5/5 [==============================] - 16s 3s/step - loss: 0.6622 - acc: 0.5620 - val_loss: 0.6797 - val_acc: 0.5450
5/5 [==============================] - 15s 3s/step - loss: 0.6614 - acc: 0.5890 - val_loss: 0.6816 - val_acc: 0.5610
5/5 [==============================] - 15s 3s/step - loss: 0.6555 - acc: 0.6220 - val_loss: 0.6646 - val_acc: 0.5900
5/5 [==============================] - 16s 3s/step - loss: 0.6492 - acc: 0.6260 - val_loss: 0.6617 - val_acc: 0.5990
5/5 [==============================] - 17s 3s/step - loss: 0.6451 - acc: 0.6400 - val_loss: 0.6661 - val_acc: 0.6210
5/5 [==============================] - 16s 3s/step - loss: 0.6328 - acc: 0.6780 - val_loss: 0.6513 - val_acc: 0.6230
5/5 [==============================] - 16s 3s/step - loss: 0.6330 - acc: 0.6620 - val_loss: 0.6709 - val_acc: 0.6170
5/5 [==============================] - 16s 3s/step - loss: 0.6411 - acc: 0.6800 - val_loss: 0.6497 - val_acc: 0.6520
5/5 [==============================] - 16s 3s/step - loss: 0.6198 - acc: 0.7080 - val_loss: 0.6462 - val_acc: 0.6500
5/5 [==============================] - 15s 3s/step - loss: 0.6180 - acc: 0.6900 - val_loss: 0.6660 - val_acc: 0.5960
5/5 [==============================] - 19s 3s/step - loss: 0.5932 - acc: 0.7030 - val_loss: 0.6177 - val_acc: 0.6730
5/5 [==============================] - 16s 3s/step - loss: 0.5936 - acc: 0.7060 - val_loss: 0.6405 - val_acc: 0.6470
5/5 [==============================] - 18s 3s/step - loss: 0.5714 - acc: 0.7210 - val_loss: 0.6253 - val_acc: 0.6700
5/5 [==============================] - 15s 3s/step - loss: 0.5607 - acc: 0.7180 - val_loss: 1.5137 - val_acc: 0.5030
5/5 [==============================] - 17s 3s/step - loss: 0.7550 - acc: 0.6220 - val_loss: 0.6440 - val_acc: 0.6470
5/5 [==============================] - 14s 3s/step - loss: 0.6001 - acc: 0.6710 - val_loss: 0.6584 - val_acc: 0.6440
5/5 [==============================] - 16s 3s/step - loss: 0.5673 - acc: 0.7400 - val_loss: 0.6596 - val_acc: 0.6210
5/5 [==============================] - 16s 3s/step - loss: 0.5539 - acc: 0.7750 - val_loss: 0.6843 - val_acc: 0.6190
5/5 [==============================] - 16s 3s/step - loss: 0.5700 - acc: 0.7200 - val_loss: 0.6497 - val_acc: 0.6220
5/5 [==============================] - 18s 4s/step - loss: 0.5586 - acc: 0.7320 - val_loss: 0.6514 - val_acc: 0.6450
5/5 [==============================] - 19s 3s/step - loss: 0.5307 - acc: 0.7580 - val_loss: 0.6429 - val_acc: 0.6580
5/5 [==============================] - 16s 3s/step - loss: 0.5265 - acc: 0.7730 - val_loss: 0.6007 - val_acc: 0.6820
5/5 [==============================] - 17s 3s/step - loss: 0.5119 - acc: 0.7670 - val_loss: 0.6399 - val_acc: 0.6490
5/5 [==============================] - 16s 3s/step - loss: 0.5077 - acc: 0.7580 - val_loss: 0.6518 - val_acc: 0.6340
5/5 [==============================] - 16s 3s/step - loss: 0.5043 - acc: 0.7720 - val_loss: 0.6606 - val_acc: 0.6600
5/5 [==============================] - 16s 3s/step - loss: 0.5030 - acc: 0.7600 - val_loss: 0.6911 - val_acc: 0.6480
5/5 [==============================] - 17s 3s/step - loss: 0.4882 - acc: 0.7780 - val_loss: 0.6404 - val_acc: 0.6340
5/5 [==============================] - 14s 3s/step - loss: 0.4983 - acc: 0.7890 - val_loss: 0.6627 - val_acc: 0.6450
5/5 [==============================] - 17s 3s/step - loss: 0.4566 - acc: 0.8110 - val_loss: 0.6635 - val_acc: 0.6550
5/5 [==============================] - 16s 3s/step - loss: 0.4773 - acc: 0.7810 - val_loss: 0.6271 - val_acc: 0.6620
5/5 [==============================] - 15s 3s/step - loss: 0.4954 - acc: 0.7670 - val_loss: 0.6678 - val_acc: 0.6780
5/5 [==============================] - 13s 3s/step - loss: 0.4498 - acc: 0.8060 - val_loss: 0.6634 - val_acc: 0.6280
5/5 [==============================] - 15s 3s/step - loss: 0.4663 - acc: 0.7900 - val_loss: 0.7176 - val_acc: 0.6110
5/5 [==============================] - 16s 3s/step - loss: 0.4632 - acc: 0.7890 - val_loss: 0.6509 - val_acc: 0.6690
5/5 [==============================] - 17s 3s/step - loss: 0.4689 - acc: 0.7910 - val_loss: 0.6928 - val_acc: 0.6560
5/5 [==============================] - 15s 3s/step - loss: 0.4516 - acc: 0.8080 - val_loss: 0.6954 - val_acc: 0.6440
5/5 [==============================] - 16s 3s/step - loss: 0.4424 - acc: 0.8070 - val_loss: 0.7394 - val_acc: 0.6290
5/5 [==============================] - 16s 3s/step - loss: 0.4221 - acc: 0.8190 - val_loss: 0.6432 - val_acc: 0.6540
5/5 [==============================] - 14s 3s/step - loss: 0.4376 - acc: 0.7970 - val_loss: 0.6937 - val_acc: 0.6200
5/5 [==============================] - 15s 3s/step - loss: 0.4465 - acc: 0.8160 - val_loss: 0.7369 - val_acc: 0.6360
5/5 [==============================] - 16s 3s/step - loss: 0.4335 - acc: 0.8030 - val_loss: 0.6853 - val_acc: 0.6310
32/32 [==============================] - 3s 84ms/step - loss: 0.6897 - acc: 0.6390
Final accuracy on test data:  0.6389999985694885
Final loss on test data:  0.689733624458313



**example 7: real faces vs stable diffusion :**
computer vision cv2 version: 4.5.5
100000 real images processed for training
2444 fake images processed for training
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 197, 197, 32)      1568      
                                                                 
 flatten (Flatten)           (None, 1241888)           0         
                                                                 
 dense (Dense)               (None, 32)                39740448  
                                                                 
 dense_1 (Dense)             (None, 2)                 66        
                                                                 
=================================================================
Total params: 39742082 (151.60 MB)
Trainable params: 39742082 (151.60 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
5/5 [==============================] - 27s 5s/step - loss: 37.8736 - acc: 0.4850 - val_loss: 21.2901 - val_acc: 0.5000
5/5 [==============================] - 23s 4s/step - loss: 13.7420 - acc: 0.5000 - val_loss: 2.5342 - val_acc: 0.5000
5/5 [==============================] - 26s 4s/step - loss: 1.0140 - acc: 0.5530 - val_loss: 0.6916 - val_acc: 0.5090
5/5 [==============================] - 29s 6s/step - loss: 0.6901 - acc: 0.5170 - val_loss: 0.6668 - val_acc: 0.5600
5/5 [==============================] - 22s 4s/step - loss: 0.6338 - acc: 0.6360 - val_loss: 0.5958 - val_acc: 0.6860
5/5 [==============================] - 25s 4s/step - loss: 0.5551 - acc: 0.7170 - val_loss: 0.5283 - val_acc: 0.7250
5/5 [==============================] - 23s 4s/step - loss: 0.4803 - acc: 0.7480 - val_loss: 0.5059 - val_acc: 0.7500
5/5 [==============================] - 29s 6s/step - loss: 0.4567 - acc: 0.7710 - val_loss: 0.4540 - val_acc: 0.7750
5/5 [==============================] - 26s 5s/step - loss: 0.4018 - acc: 0.8180 - val_loss: 0.4280 - val_acc: 0.7960
5/5 [==============================] - 27s 4s/step - loss: 0.3881 - acc: 0.8180 - val_loss: 0.4191 - val_acc: 0.7990
5/5 [==============================] - 22s 4s/step - loss: 0.3972 - acc: 0.8170 - val_loss: 0.4272 - val_acc: 0.7920
5/5 [==============================] - 35s 6s/step - loss: 0.3563 - acc: 0.8360 - val_loss: 0.4062 - val_acc: 0.8050
5/5 [==============================] - 22s 4s/step - loss: 0.3404 - acc: 0.8440 - val_loss: 0.3723 - val_acc: 0.8310
5/5 [==============================] - 21s 4s/step - loss: 0.3303 - acc: 0.8730 - val_loss: 0.3522 - val_acc: 0.8520
5/5 [==============================] - 19s 4s/step - loss: 0.3265 - acc: 0.8530 - val_loss: 0.3906 - val_acc: 0.8270
5/5 [==============================] - 19s 4s/step - loss: 0.3245 - acc: 0.8590 - val_loss: 0.3341 - val_acc: 0.8510
5/5 [==============================] - 17s 3s/step - loss: 0.2832 - acc: 0.8730 - val_loss: 0.4044 - val_acc: 0.8170
5/5 [==============================] - 32s 7s/step - loss: 0.3247 - acc: 0.8540 - val_loss: 0.3202 - val_acc: 0.8530
5/5 [==============================] - 24s 5s/step - loss: 0.3003 - acc: 0.8690 - val_loss: 0.3887 - val_acc: 0.8300
5/5 [==============================] - 29s 5s/step - loss: 0.2841 - acc: 0.8680 - val_loss: 0.3171 - val_acc: 0.8520
5/5 [==============================] - 25s 4s/step - loss: 0.2507 - acc: 0.8890 - val_loss: 0.3412 - val_acc: 0.8410
5/5 [==============================] - 21s 4s/step - loss: 0.2498 - acc: 0.9010 - val_loss: 0.3370 - val_acc: 0.8540
5/5 [==============================] - 23s 4s/step - loss: 0.2522 - acc: 0.8880 - val_loss: 0.2997 - val_acc: 0.8770
5/5 [==============================] - 27s 5s/step - loss: 0.2373 - acc: 0.9080 - val_loss: 0.3764 - val_acc: 0.8500
5/5 [==============================] - 24s 4s/step - loss: 0.2256 - acc: 0.9040 - val_loss: 0.3316 - val_acc: 0.8600
5/5 [==============================] - 26s 5s/step - loss: 0.2585 - acc: 0.8890 - val_loss: 0.3124 - val_acc: 0.8600
5/5 [==============================] - 20s 3s/step - loss: 0.2525 - acc: 0.9100 - val_loss: 0.3423 - val_acc: 0.8540
5/5 [==============================] - 39s 7s/step - loss: 0.2542 - acc: 0.8900 - val_loss: 0.3556 - val_acc: 0.8360
5/5 [==============================] - 29s 5s/step - loss: 0.2019 - acc: 0.9180 - val_loss: 0.3848 - val_acc: 0.8550
5/5 [==============================] - 28s 5s/step - loss: 0.1909 - acc: 0.9190 - val_loss: 0.3356 - val_acc: 0.8510
5/5 [==============================] - 28s 5s/step - loss: 0.1865 - acc: 0.9190 - val_loss: 0.3347 - val_acc: 0.8630
5/5 [==============================] - 33s 6s/step - loss: 0.1563 - acc: 0.9400 - val_loss: 0.2825 - val_acc: 0.8750
5/5 [==============================] - 24s 4s/step - loss: 0.1654 - acc: 0.9320 - val_loss: 0.2690 - val_acc: 0.8890
5/5 [==============================] - 28s 5s/step - loss: 0.1672 - acc: 0.9260 - val_loss: 0.2819 - val_acc: 0.8730
5/5 [==============================] - 20s 4s/step - loss: 0.1674 - acc: 0.9370 - val_loss: 0.3481 - val_acc: 0.8560
5/5 [==============================] - 27s 4s/step - loss: 0.1643 - acc: 0.9400 - val_loss: 0.3648 - val_acc: 0.8560
5/5 [==============================] - 22s 4s/step - loss: 0.1413 - acc: 0.9540 - val_loss: 0.3069 - val_acc: 0.8780
5/5 [==============================] - 19s 3s/step - loss: 0.1402 - acc: 0.9460 - val_loss: 0.2772 - val_acc: 0.8840
5/5 [==============================] - 33s 5s/step - loss: 0.1585 - acc: 0.9430 - val_loss: 0.4090 - val_acc: 0.8410
5/5 [==============================] - 25s 4s/step - loss: 0.1586 - acc: 0.9500 - val_loss: 0.3140 - val_acc: 0.8710
5/5 [==============================] - 22s 4s/step - loss: 0.1376 - acc: 0.9580 - val_loss: 0.3481 - val_acc: 0.8600
5/5 [==============================] - 21s 4s/step - loss: 0.1256 - acc: 0.9600 - val_loss: 0.3346 - val_acc: 0.8740
5/5 [==============================] - 20s 4s/step - loss: 0.1128 - acc: 0.9640 - val_loss: 0.3727 - val_acc: 0.8620
5/5 [==============================] - 22s 4s/step - loss: 0.1060 - acc: 0.9660 - val_loss: 0.4765 - val_acc: 0.8330
5/5 [==============================] - 22s 4s/step - loss: 0.1544 - acc: 0.9410 - val_loss: 0.4239 - val_acc: 0.8390
5/5 [==============================] - 21s 4s/step - loss: 0.1529 - acc: 0.9430 - val_loss: 0.3091 - val_acc: 0.8680
5/5 [==============================] - 20s 4s/step - loss: 0.1317 - acc: 0.9540 - val_loss: 0.3460 - val_acc: 0.8690
5/5 [==============================] - 23s 4s/step - loss: 0.1150 - acc: 0.9630 - val_loss: 0.5137 - val_acc: 0.8380
5/5 [==============================] - 23s 4s/step - loss: 0.0952 - acc: 0.9660 - val_loss: 0.2897 - val_acc: 0.8870
5/5 [==============================] - 20s 4s/step - loss: 0.1175 - acc: 0.9630 - val_loss: 0.4895 - val_acc: 0.8450
29/29 [==============================] - 4s 91ms/step - loss: 0.4856 - acc: 0.8611
Final accuracy on test data:  0.8611111044883728
Final loss on test data:  0.4855976700782776



**example 8: real faces vs star gan :**
computer vision cv2 version: 4.5.5
100000 real images processed for training
9995 fake images processed for training
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 197, 197, 32)      1568      
                                                                 
 flatten (Flatten)           (None, 1241888)           0         
                                                                 
 dense (Dense)               (None, 32)                39740448  
                                                                 
 dense_1 (Dense)             (None, 2)                 66        
                                                                 
=================================================================
Total params: 39742082 (151.60 MB)
Trainable params: 39742082 (151.60 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
5/5 [==============================] - 23s 4s/step - loss: 9.1586 - acc: 0.5260 - val_loss: 1.7394 - val_acc: 0.5090
5/5 [==============================] - 20s 4s/step - loss: 1.3067 - acc: 0.5380 - val_loss: 1.1058 - val_acc: 0.5010
5/5 [==============================] - 27s 5s/step - loss: 1.1045 - acc: 0.5010 - val_loss: 0.6809 - val_acc: 0.5390
5/5 [==============================] - 23s 4s/step - loss: 0.8726 - acc: 0.5170 - val_loss: 0.8547 - val_acc: 0.5030
5/5 [==============================] - 23s 4s/step - loss: 0.7291 - acc: 0.5400 - val_loss: 0.6377 - val_acc: 0.6950
5/5 [==============================] - 19s 4s/step - loss: 0.6532 - acc: 0.5970 - val_loss: 0.6603 - val_acc: 0.5450
5/5 [==============================] - 20s 4s/step - loss: 0.6345 - acc: 0.6320 - val_loss: 0.6227 - val_acc: 0.6980
5/5 [==============================] - 31s 6s/step - loss: 0.6006 - acc: 0.7100 - val_loss: 0.5800 - val_acc: 0.7450
5/5 [==============================] - 19s 4s/step - loss: 0.5965 - acc: 0.7140 - val_loss: 0.5276 - val_acc: 0.7570
5/5 [==============================] - 19s 4s/step - loss: 0.5310 - acc: 0.7480 - val_loss: 0.5839 - val_acc: 0.6850
5/5 [==============================] - 18s 3s/step - loss: 0.5274 - acc: 0.7570 - val_loss: 0.5200 - val_acc: 0.7560
5/5 [==============================] - 21s 4s/step - loss: 0.5048 - acc: 0.7700 - val_loss: 0.4606 - val_acc: 0.7950
5/5 [==============================] - 21s 4s/step - loss: 0.4472 - acc: 0.8100 - val_loss: 0.4448 - val_acc: 0.7960
5/5 [==============================] - 21s 4s/step - loss: 0.4082 - acc: 0.8320 - val_loss: 0.3850 - val_acc: 0.8270
5/5 [==============================] - 20s 4s/step - loss: 0.4094 - acc: 0.8180 - val_loss: 0.4106 - val_acc: 0.8290
5/5 [==============================] - 19s 3s/step - loss: 0.3868 - acc: 0.8330 - val_loss: 0.4564 - val_acc: 0.7840
5/5 [==============================] - 19s 3s/step - loss: 0.4132 - acc: 0.8210 - val_loss: 0.4142 - val_acc: 0.8360
5/5 [==============================] - 20s 4s/step - loss: 0.4289 - acc: 0.8150 - val_loss: 0.3902 - val_acc: 0.8520
5/5 [==============================] - 15s 3s/step - loss: 0.4153 - acc: 0.8230 - val_loss: 0.4399 - val_acc: 0.8110
5/5 [==============================] - 18s 3s/step - loss: 0.3298 - acc: 0.8580 - val_loss: 0.3987 - val_acc: 0.8230
5/5 [==============================] - 18s 3s/step - loss: 0.3792 - acc: 0.8400 - val_loss: 0.4116 - val_acc: 0.8020
5/5 [==============================] - 20s 4s/step - loss: 0.3575 - acc: 0.8420 - val_loss: 0.3863 - val_acc: 0.8330
5/5 [==============================] - 25s 4s/step - loss: 0.3481 - acc: 0.8550 - val_loss: 0.3594 - val_acc: 0.8390
5/5 [==============================] - 20s 3s/step - loss: 0.3221 - acc: 0.8690 - val_loss: 0.3536 - val_acc: 0.8360
5/5 [==============================] - 18s 3s/step - loss: 0.3175 - acc: 0.8700 - val_loss: 0.3368 - val_acc: 0.8690
5/5 [==============================] - 20s 4s/step - loss: 0.2753 - acc: 0.9010 - val_loss: 0.3444 - val_acc: 0.8560
5/5 [==============================] - 17s 3s/step - loss: 0.3200 - acc: 0.8750 - val_loss: 0.3569 - val_acc: 0.8530
5/5 [==============================] - 19s 3s/step - loss: 0.2755 - acc: 0.8880 - val_loss: 0.3442 - val_acc: 0.8460
5/5 [==============================] - 18s 3s/step - loss: 0.2760 - acc: 0.8860 - val_loss: 0.3100 - val_acc: 0.8790
5/5 [==============================] - 18s 4s/step - loss: 0.2832 - acc: 0.8810 - val_loss: 0.2904 - val_acc: 0.8860
5/5 [==============================] - 20s 4s/step - loss: 0.2384 - acc: 0.9030 - val_loss: 0.2715 - val_acc: 0.8960
5/5 [==============================] - 12s 3s/step - loss: 0.2150 - acc: 0.9180 - val_loss: 0.2632 - val_acc: 0.9000
5/5 [==============================] - 15s 3s/step - loss: 0.2908 - acc: 0.8710 - val_loss: 0.2963 - val_acc: 0.8680
5/5 [==============================] - 18s 3s/step - loss: 0.2385 - acc: 0.8960 - val_loss: 0.3017 - val_acc: 0.8760
5/5 [==============================] - 15s 3s/step - loss: 0.2648 - acc: 0.8920 - val_loss: 0.3312 - val_acc: 0.8480
5/5 [==============================] - 17s 3s/step - loss: 0.2818 - acc: 0.8800 - val_loss: 0.3013 - val_acc: 0.8820
5/5 [==============================] - 11s 2s/step - loss: 0.2464 - acc: 0.9100 - val_loss: 0.2873 - val_acc: 0.8900
5/5 [==============================] - 14s 3s/step - loss: 0.2252 - acc: 0.9190 - val_loss: 0.2805 - val_acc: 0.8860
5/5 [==============================] - 16s 3s/step - loss: 0.2119 - acc: 0.9260 - val_loss: 0.2722 - val_acc: 0.8860
5/5 [==============================] - 15s 3s/step - loss: 0.2435 - acc: 0.8970 - val_loss: 0.2977 - val_acc: 0.8910
5/5 [==============================] - 16s 3s/step - loss: 0.2381 - acc: 0.9200 - val_loss: 0.3416 - val_acc: 0.8470
5/5 [==============================] - 14s 3s/step - loss: 0.2292 - acc: 0.9080 - val_loss: 0.2679 - val_acc: 0.8900
5/5 [==============================] - 15s 3s/step - loss: 0.2240 - acc: 0.9110 - val_loss: 0.2925 - val_acc: 0.8740
5/5 [==============================] - 23s 4s/step - loss: 0.2029 - acc: 0.9250 - val_loss: 0.2643 - val_acc: 0.8920
5/5 [==============================] - 17s 3s/step - loss: 0.1999 - acc: 0.9250 - val_loss: 0.3225 - val_acc: 0.8650
5/5 [==============================] - 18s 3s/step - loss: 0.2685 - acc: 0.8930 - val_loss: 0.2559 - val_acc: 0.9070
5/5 [==============================] - 15s 3s/step - loss: 0.2360 - acc: 0.9180 - val_loss: 0.2335 - val_acc: 0.9060
5/5 [==============================] - 15s 3s/step - loss: 0.2414 - acc: 0.9070 - val_loss: 0.2599 - val_acc: 0.8980
5/5 [==============================] - 14s 3s/step - loss: 0.2207 - acc: 0.9110 - val_loss: 0.2783 - val_acc: 0.8820
5/5 [==============================] - 16s 3s/step - loss: 0.2121 - acc: 0.9240 - val_loss: 0.2643 - val_acc: 0.8900
32/32 [==============================] - 3s 84ms/step - loss: 0.2584 - acc: 0.8970
Final accuracy on test data:  0.8970000147819519
Final loss on test data:  0.25837790966033936



**example 9: real faces vs styleGAN :**
computer vision cv2 version: 4.5.5
100000 real images processed for training
10000 fake images processed for training
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 197, 197, 32)      1568      
                                                                 
 flatten (Flatten)           (None, 1241888)           0         
                                                                 
 dense (Dense)               (None, 32)                39740448  
                                                                 
 dense_1 (Dense)             (None, 2)                 66        
                                                                 
=================================================================
Total params: 39742082 (151.60 MB)
Trainable params: 39742082 (151.60 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
3/3 [==============================] - 12s 3s/step - loss: 35.7663 - acc: 0.8234 - val_loss: 38.9415 - val_acc: 0.9225
3/3 [==============================] - 12s 3s/step - loss: 29.9292 - acc: 0.9294 - val_loss: 18.2157 - val_acc: 0.9276
3/3 [==============================] - 10s 3s/step - loss: 15.0590 - acc: 0.9259 - val_loss: 5.3784 - val_acc: 0.9225
3/3 [==============================] - 11s 3s/step - loss: 3.8548 - acc: 0.9141 - val_loss: 0.4020 - val_acc: 0.9241
3/3 [==============================] - 12s 3s/step - loss: 0.5911 - acc: 0.4173 - val_loss: 0.6932 - val_acc: 0.0876
3/3 [==============================] - 10s 3s/step - loss: 0.6927 - acc: 0.6208 - val_loss: 0.6916 - val_acc: 0.9158
3/3 [==============================] - 8s 2s/step - loss: 0.6909 - acc: 0.9363 - val_loss: 0.6896 - val_acc: 0.8961
3/3 [==============================] - 9s 3s/step - loss: 0.6888 - acc: 0.9009 - val_loss: 0.6870 - val_acc: 0.9158
3/3 [==============================] - 8s 3s/step - loss: 0.6864 - acc: 0.9074 - val_loss: 0.6846 - val_acc: 0.9074
3/3 [==============================] - 10s 3s/step - loss: 0.6838 - acc: 0.9107 - val_loss: 0.6819 - val_acc: 0.9124
3/3 [==============================] - 7s 2s/step - loss: 0.6809 - acc: 0.9191 - val_loss: 0.6795 - val_acc: 0.9025
3/3 [==============================] - 8s 3s/step - loss: 0.6787 - acc: 0.9042 - val_loss: 0.6773 - val_acc: 0.8929
3/3 [==============================] - 9s 3s/step - loss: 0.6755 - acc: 0.9158 - val_loss: 0.6737 - val_acc: 0.9141
3/3 [==============================] - 9s 3s/step - loss: 0.6737 - acc: 0.8977 - val_loss: 0.6714 - val_acc: 0.9074
3/3 [==============================] - 7s 3s/step - loss: 0.6707 - acc: 0.9058 - val_loss: 0.6685 - val_acc: 0.9124
3/3 [==============================] - 8s 3s/step - loss: 0.6676 - acc: 0.9058 - val_loss: 0.6659 - val_acc: 0.9124
3/3 [==============================] - 8s 3s/step - loss: 0.6643 - acc: 0.9174 - val_loss: 0.6634 - val_acc: 0.9058
3/3 [==============================] - 13s 5s/step - loss: 0.6605 - acc: 0.9381 - val_loss: 0.6596 - val_acc: 0.9259
3/3 [==============================] - 10s 3s/step - loss: 0.6602 - acc: 0.9042 - val_loss: 0.6575 - val_acc: 0.9158
3/3 [==============================] - 10s 3s/step - loss: 0.6575 - acc: 0.9107 - val_loss: 0.6556 - val_acc: 0.9107
3/3 [==============================] - 12s 3s/step - loss: 0.6523 - acc: 0.9242 - val_loss: 0.6530 - val_acc: 0.9124
3/3 [==============================] - 14s 4s/step - loss: 0.6506 - acc: 0.9091 - val_loss: 0.6481 - val_acc: 0.9346
3/3 [==============================] - 11s 3s/step - loss: 0.6464 - acc: 0.9174 - val_loss: 0.6497 - val_acc: 0.8977
3/3 [==============================] - 9s 3s/step - loss: 0.6441 - acc: 0.9294 - val_loss: 0.6436 - val_acc: 0.9091
3/3 [==============================] - 18s 5s/step - loss: 0.6436 - acc: 0.9058 - val_loss: 0.6402 - val_acc: 0.9328
3/3 [==============================] - 10s 3s/step - loss: 0.6379 - acc: 0.9191 - val_loss: 0.6414 - val_acc: 0.8881
3/3 [==============================] - 7s 2s/step - loss: 0.6387 - acc: 0.9174 - val_loss: 0.6355 - val_acc: 0.8993
3/3 [==============================] - 9s 3s/step - loss: 0.6362 - acc: 0.8897 - val_loss: 0.6288 - val_acc: 0.9058
3/3 [==============================] - 12s 5s/step - loss: 0.6237 - acc: 0.9346 - val_loss: 0.6230 - val_acc: 0.9191
3/3 [==============================] - 11s 4s/step - loss: 0.6213 - acc: 0.8913 - val_loss: 0.5969 - val_acc: 0.8977
3/3 [==============================] - 9s 3s/step - loss: 0.6183 - acc: 0.9191 - val_loss: 0.4338 - val_acc: 0.9074
3/3 [==============================] - 13s 5s/step - loss: 0.4959 - acc: 0.8993 - val_loss: 0.4680 - val_acc: 0.9091
3/3 [==============================] - 11s 3s/step - loss: 0.5377 - acc: 0.9074 - val_loss: 0.4795 - val_acc: 0.9091
3/3 [==============================] - 8s 3s/step - loss: 0.5207 - acc: 0.9294 - val_loss: 0.4481 - val_acc: 0.9328
3/3 [==============================] - 10s 3s/step - loss: 0.5515 - acc: 0.8913 - val_loss: 0.4212 - val_acc: 0.9207
3/3 [==============================] - 9s 3s/step - loss: 0.5224 - acc: 0.8969 - val_loss: 0.4032 - val_acc: 0.8977
3/3 [==============================] - 12s 3s/step - loss: 0.3616 - acc: 0.9158 - val_loss: 0.3631 - val_acc: 0.9107
3/3 [==============================] - 10s 3s/step - loss: 0.3002 - acc: 0.9196 - val_loss: 0.3423 - val_acc: 0.9103
3/3 [==============================] - 10s 3s/step - loss: 0.3324 - acc: 0.9174 - val_loss: 0.3725 - val_acc: 0.8809
3/3 [==============================] - 9s 3s/step - loss: 0.4015 - acc: 0.8651 - val_loss: 0.3333 - val_acc: 0.9124
3/3 [==============================] - 8s 3s/step - loss: 0.3304 - acc: 0.9124 - val_loss: 0.3460 - val_acc: 0.9024
3/3 [==============================] - 12s 4s/step - loss: 0.3260 - acc: 0.9107 - val_loss: 0.3417 - val_acc: 0.9009
3/3 [==============================] - 11s 3s/step - loss: 0.3052 - acc: 0.9124 - val_loss: 0.3243 - val_acc: 0.9025
3/3 [==============================] - 12s 4s/step - loss: 0.2342 - acc: 0.9346 - val_loss: 0.3256 - val_acc: 0.9107
3/3 [==============================] - 11s 4s/step - loss: 0.2744 - acc: 0.9174 - val_loss: 0.2940 - val_acc: 0.9191
3/3 [==============================] - 15s 4s/step - loss: 0.3056 - acc: 0.9107 - val_loss: 0.2886 - val_acc: 0.9156
3/3 [==============================] - 10s 3s/step - loss: 0.3450 - acc: 0.8881 - val_loss: 0.3259 - val_acc: 0.9091
3/3 [==============================] - 11s 3s/step - loss: 0.3483 - acc: 0.8879 - val_loss: 0.3087 - val_acc: 0.9121
3/3 [==============================] - 8s 2s/step - loss: 0.3057 - acc: 0.9109 - val_loss: 0.2825 - val_acc: 0.9174
3/3 [==============================] - 13s 4s/step - loss: 0.3236 - acc: 0.9089 - val_loss: 0.3462 - val_acc: 0.8959
18/18 [==============================] - 2s 80ms/step - loss: 0.3124 - acc: 0.9091
Final accuracy on test data:  0.9090909361839294
Final loss on test data:  0.31242069602012634



**example 10: real faces vs all models combined :**
computer vision cv2 version: 4.5.5
100000 real images processed for training
57946 fake images processed for training
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 197, 197, 32)      1568      
                                                                 
 flatten (Flatten)           (None, 1241888)           0         
                                                                 
 dense (Dense)               (None, 32)                39740448  
                                                                 
 dense_1 (Dense)             (None, 2)                 66        
                                                                 
=================================================================
Total params: 39742082 (151.60 MB)
Trainable params: 39742082 (151.60 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
4/4 [==============================] - 16s 4s/step - loss: 30.1810 - acc: 0.5192 - val_loss: 4.4125 - val_acc: 0.3742
5/5 [==============================] - 16s 3s/step - loss: 4.2987 - acc: 0.4722 - val_loss: 4.4430 - val_acc: 0.6519
4/4 [==============================] - 14s 4s/step - loss: 2.6341 - acc: 0.6279 - val_loss: 2.1925 - val_acc: 0.3947
4/4 [==============================] - 17s 4s/step - loss: 1.6963 - acc: 0.4278 - val_loss: 1.3179 - val_acc: 0.6143
4/4 [==============================] - 18s 3s/step - loss: 1.3071 - acc: 0.6545 - val_loss: 0.8823 - val_acc: 0.6456
4/4 [==============================] - 15s 4s/step - loss: 0.8313 - acc: 0.5425 - val_loss: 0.9462 - val_acc: 0.4428
4/4 [==============================] - 14s 3s/step - loss: 0.7763 - acc: 0.5297 - val_loss: 0.6825 - val_acc: 0.6376
4/4 [==============================] - 13s 3s/step - loss: 0.7389 - acc: 0.6562 - val_loss: 0.7013 - val_acc: 0.6592
4/4 [==============================] - 13s 3s/step - loss: 0.6677 - acc: 0.6584 - val_loss: 0.6419 - val_acc: 0.6539
5/5 [==============================] - 15s 3s/step - loss: 0.6719 - acc: 0.5916 - val_loss: 0.6669 - val_acc: 0.5882
4/4 [==============================] - 14s 3s/step - loss: 0.6490 - acc: 0.6263 - val_loss: 0.6316 - val_acc: 0.6462
4/4 [==============================] - 12s 3s/step - loss: 0.6622 - acc: 0.6361 - val_loss: 0.6282 - val_acc: 0.6680
4/4 [==============================] - 14s 4s/step - loss: 0.6343 - acc: 0.6562 - val_loss: 0.6511 - val_acc: 0.6330
4/4 [==============================] - 12s 3s/step - loss: 0.6364 - acc: 0.6658 - val_loss: 0.6460 - val_acc: 0.6488
5/5 [==============================] - 14s 3s/step - loss: 0.6453 - acc: 0.6344 - val_loss: 0.6456 - val_acc: 0.6250
4/4 [==============================] - 12s 3s/step - loss: 0.6328 - acc: 0.6564 - val_loss: 0.6276 - val_acc: 0.6467
5/5 [==============================] - 28s 6s/step - loss: 0.6477 - acc: 0.6299 - val_loss: 0.6284 - val_acc: 0.6679
4/4 [==============================] - 26s 7s/step - loss: 0.6164 - acc: 0.6744 - val_loss: 0.6374 - val_acc: 0.6521
5/5 [==============================] - 28s 5s/step - loss: 0.6225 - acc: 0.6498 - val_loss: 0.6438 - val_acc: 0.6395
4/4 [==============================] - 26s 6s/step - loss: 0.6423 - acc: 0.6624 - val_loss: 0.6311 - val_acc: 0.6572
4/4 [==============================] - 28s 7s/step - loss: 0.6016 - acc: 0.6738 - val_loss: 0.6183 - val_acc: 0.6739
5/5 [==============================] - 25s 5s/step - loss: 0.6378 - acc: 0.6498 - val_loss: 0.6188 - val_acc: 0.6747
4/4 [==============================] - 15s 3s/step - loss: 0.6068 - acc: 0.6975 - val_loss: 0.6244 - val_acc: 0.6667
5/5 [==============================] - 15s 3s/step - loss: 0.6475 - acc: 0.6282 - val_loss: 0.6250 - val_acc: 0.6574
4/4 [==============================] - 17s 4s/step - loss: 0.6112 - acc: 0.6822 - val_loss: 0.6305 - val_acc: 0.6466
4/4 [==============================] - 19s 4s/step - loss: 0.6204 - acc: 0.6790 - val_loss: 0.6150 - val_acc: 0.6756
5/5 [==============================] - 19s 3s/step - loss: 0.6308 - acc: 0.6534 - val_loss: 0.6060 - val_acc: 0.6856
4/4 [==============================] - 17s 4s/step - loss: 0.6097 - acc: 0.7150 - val_loss: 0.6359 - val_acc: 0.6453
4/4 [==============================] - 21s 5s/step - loss: 0.6042 - acc: 0.6907 - val_loss: 0.6163 - val_acc: 0.6802
4/4 [==============================] - 24s 5s/step - loss: 0.5850 - acc: 0.6988 - val_loss: 0.6396 - val_acc: 0.6423
4/4 [==============================] - 17s 4s/step - loss: 0.6046 - acc: 0.6732 - val_loss: 0.6162 - val_acc: 0.6683
5/5 [==============================] - 18s 3s/step - loss: 0.6118 - acc: 0.6592 - val_loss: 0.6092 - val_acc: 0.6854
4/4 [==============================] - 16s 4s/step - loss: 0.5944 - acc: 0.6848 - val_loss: 0.6099 - val_acc: 0.6752
4/4 [==============================] - 19s 4s/step - loss: 0.5961 - acc: 0.7069 - val_loss: 0.6362 - val_acc: 0.6526
5/5 [==============================] - 13s 3s/step - loss: 0.6214 - acc: 0.6695 - val_loss: 0.6263 - val_acc: 0.6568
5/5 [==============================] - 16s 3s/step - loss: 0.6315 - acc: 0.6482 - val_loss: 0.6153 - val_acc: 0.6675
4/4 [==============================] - 23s 6s/step - loss: 0.6197 - acc: 0.6594 - val_loss: 0.6141 - val_acc: 0.6815
4/4 [==============================] - 18s 4s/step - loss: 0.5859 - acc: 0.7115 - val_loss: 0.6067 - val_acc: 0.6738
4/4 [==============================] - 14s 3s/step - loss: 0.5843 - acc: 0.6955 - val_loss: 0.5857 - val_acc: 0.7050
5/5 [==============================] - 15s 3s/step - loss: 0.5879 - acc: 0.6877 - val_loss: 0.6006 - val_acc: 0.6996
4/4 [==============================] - 15s 4s/step - loss: 0.6138 - acc: 0.6800 - val_loss: 0.6179 - val_acc: 0.6839
4/4 [==============================] - 13s 3s/step - loss: 0.6069 - acc: 0.7003 - val_loss: 0.6145 - val_acc: 0.6675
5/5 [==============================] - 16s 3s/step - loss: 0.6206 - acc: 0.6692 - val_loss: 0.6304 - val_acc: 0.6607
4/4 [==============================] - 13s 3s/step - loss: 0.6035 - acc: 0.6826 - val_loss: 0.6244 - val_acc: 0.6693
4/4 [==============================] - 14s 3s/step - loss: 0.5881 - acc: 0.6943 - val_loss: 0.6153 - val_acc: 0.6751
4/4 [==============================] - 16s 3s/step - loss: 0.6180 - acc: 0.6796 - val_loss: 0.5925 - val_acc: 0.7012
4/4 [==============================] - 13s 3s/step - loss: 0.5862 - acc: 0.6878 - val_loss: 0.5962 - val_acc: 0.6924
4/4 [==============================] - 12s 3s/step - loss: 0.5846 - acc: 0.7005 - val_loss: 0.6038 - val_acc: 0.6783
4/4 [==============================] - 12s 3s/step - loss: 0.5774 - acc: 0.7184 - val_loss: 0.5765 - val_acc: 0.7135
4/4 [==============================] - 11s 3s/step - loss: 0.6044 - acc: 0.6763 - val_loss: 0.5846 - val_acc: 0.7132
25/25 [==============================] - 2s 83ms/step - loss: 0.5898 - acc: 0.7095
Final accuracy on test data:  0.7095115780830383
Final loss on test data:  0.5897865891456604


**How it works**
A convolutional neural network (CNN) uses a series of convolutional layers to process images and make decisions, like distinguishing between real and AI-generated faces. Here's a detailed breakdown of how this works:

1. **Convolutional Layers:** The core component of a CNN, these layers apply a set of filters (kernels) to the input images. Each filter is a small matrix that slides over the image, performing an element-wise multiplication with each overlapping region, then summing the results. This produces a new feature map (or activation map) that highlights certain aspects of the input, like edges, textures, or shapes.

   - **Feature Detection:** Initially, convolutional layers detect basic patterns. For example, the first layer might capture edges or simple textures. Subsequent layers, by processing these feature maps, can identify more complex features, like facial structures or textures characteristic of AI-generated images.

   - **Stride and Padding:** The stride determines how much the filter shifts after each operation. Padding (adding extra pixels around the edges) can preserve the spatial dimensions of the image throughout the convolutional operations, which is useful to maintain structural integrity.

2. **Pooling Layers:** After convolutional layers, pooling layers reduce the spatial dimensions of the feature maps. Max-pooling or average-pooling selects the maximum or average value from a region of the feature map, respectively. This reduces the computational load and helps the network generalize by focusing on key features.

3. **Non-linearity with Activation Functions:** After each convolutional layer, an activation function (often ReLU, or Rectified Linear Unit) is applied. This introduces non-linearity, allowing the network to learn complex relationships between features. The ReLU function sets all negative values in the feature map to zero, which helps to alleviate the vanishing gradient problem and accelerates training.

4. **Stacking and Depth:** As convolutional layers are stacked, the network gains depth, allowing it to detect increasingly abstract features. For instance, in face classification, early layers might detect simple features like edges or textures, while deeper layers recognize more complex patterns, like eyes, noses, or even subtle textures unique to AI-generated images.

5. **Fully Connected Layers:** After a series of convolutional and pooling layers, the final feature maps are flattened into a single vector, which is then fed into one or more fully connected layers. These layers act like a traditional neural network, learning the relationships between the flattened features and the output classes (real or AI-generated face).

6. **Output Layer:** The final fully connected layer maps the features to the desired output. In a binary classification task (such as distinguishing real from AI-generated faces), this layer uses a sigmoid function to produce a probability between 0 and 1, indicating the likelihood of the image belonging to one class or the other.

7. **Training and Backpropagation:** During training, the output of the CNN is compared to the true label using a loss function like binary cross-entropy. The network's weights are then adjusted through backpropagation, where the loss gradients are propagated backward through the network. This process optimizes the weights iteratively, improving the model's ability to distinguish between real and AI-generated faces.

Here are the steps I followed to execute this:

1. **Dataset Collection:**  This dataset consist of two categories: real human face images and AI-generated face images (from GANs or other generative models). The images need to be labeled accordingly to serve as training data.

2. **Data Preprocessing:** This involves resizing images to a consistent resolution, normalizing pixel values, and possibly augmenting the data to introduce variations (such as rotations, flips, or adjustments in brightness and contrast). This helps to improve the model's generalization.

3. **CNN Architecture:** The model architecture consists of multiple convolutional layers that learn hierarchical features from the images. Initial layers might detect basic patterns such as edges or textures, while deeper layers capture more complex structures specific to faces or their generated counterparts.

4. **Loss Function:** I used for categorical cross entropy for this binary classification task. It compares the predicted probability (of an image being real or generated) against the true label, and calculates the loss to guide the model's learning.

5. **Training Process:** During training, the dataset is split into training and validation sets. The model learns by adjusting its weights iteratively over multiple epochs. This is achieved through a process known as backpropagation, where the loss gradients are propagated back through the network to optimize weights.

6. **Evaluation:** After training, the model's performance is evaluated on a separate test set. Metrics such as accuracy, precision, recall, and F1 score are used to assess its effectiveness at distinguishing between real and AI-generated faces.

7. **Fine-tuning:** Based on the model's performance, further adjustments, such as changing hyperparameters (learning rate, batch size, or number of epochs) or altering the network architecture, can be made. Additionally, more data might be added or preprocessed differently to enhance accuracy.

8. **Deployment:** Once the model achieves satisfactory performance, it can be deployed for real-world applications. This might involve further optimizing its speed and integrating it into a larger system for automated face detection and classification.


Please see the visualization PDF to see how these layers attempt to filter images to recognize pixel patterns.

**Applying the Model to real world usage**
You can run the sample_usage.py file on the saved images to see the saved models in action. I randomly added fake and real faces into the folder "sample_images" so the user can see how the various models apply their weights to process and image and come to a consensus on the automated decision.



