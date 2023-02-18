# Neural_Network_For_Speech_Recognition

### Second Take on First Speech Technology Project - recognition of numbers with neural network.

### Description of research

1. Implementation of pre-processing --> Random neural network

### Build:
1. Conv2D [16, 3x3]                                                                     
2. Max_pooling2D [2x2]                                                            
3. Flatten                                                                 
4. Activation - ReLU
5. Dense - 10
6. Activation - Softmax [10]

Convolution and max_pool added because of "image" based approach.

### Effectiveness on evaluation data:
### App. 30%

2. Adding proper convolution --> double Conv2D - MaxPool layers

### Build:
1. Conv2D [32, 6x6]                                                                     
2. Max_pooling2D [2x2]
3. Conv2D [16, 3x3]                                                                     
4. Max_pooling2D [2x2]                                                              
5. Flatten                                                                 
6. Activation - ReLU
7. Dense - 128
9. Activation - ReLU
10. Dense - 10
11. Activation - Softmax [10]

Also added additional Dense layer for better performance.

### Effectiveness on evaluation data:
### App. 40%

3. Testing Dropout

### Build:
Added dropout layer (0.5) between Dense(128) and Dense(10).
Tested on window_length = 1024.

Tested in terms of over-train and performance.

### Effectiveness on evaluation data:
### App. 53%

### DISCLAIMER - all the builds shown above worked on MFCC (13) with ADAM optimizer at lr=0.001.

4. Adding CMVN normalization

### Build:
Added CMVN in pre-processing.
### DISCLAIMER - final version do not have this feature.

Using version from SpeechPy.

### Effectiveness on evaluation data:
### App. 57%

5. Adding another Dense layer and less dropout.

### Build:
Adding the same Dense layer and less dropout.

### Effectiveness on evaluation data:
### Dense : App. 54%
### Lower Dropout (0.3) : App. 55%

6. Adding Cross-Validation (tested on model from point 04).

### Build:
Unchanged from point 04.
Tested on 100 epoch's with 11 fold's of Cross-Validation.

### Effectiveness on evaluation data:
### App. 60%

7. Testing adding delta features to MFCC.

### Build:
Added Delta features.
Tested with and without CMVN.

### Effectiveness on evaluation data:
### CMVN : App. 58%
### No CMVN : App. 65%

Also tested different epoch's sizes and implemented EarlyStopping.

### DISCLAIMER - After this one the CMVN was deleted.

8. Tested different DCT-Types on same settings.

### Effectiveness on evaluation data:
### DCT-1 : 62.5%
### DCT-2 : 61.06%
### DCT-3 : 59.85%

9. Tested another features.

Tested:
1. Double-trained model.
2. Delta-delta MFCC
3. Different epoch's.
4. Doubling the filters in Conv2D.
5. Noise Deletion in recordings.

### Results:
No noticeable difference.

10. Added strides to Convolution2D layer.   

In comprehension with used solutions like AlexNet decided to try strides.

### Build:
Added strides = 3 to first Conv2D.

### Effectiveness on evaluation data:
### App. 53%

11. Added double strides to Convolution2D layer.   

### Build:
Added strides = 1 to first and second Conv2D.

### Effectiveness on evaluation data:
### App. 62%

### And:

### Build:
Added strides = 1 to first and strides = 2 to second Conv2D.

### Effectiveness on evaluation data:
### App. 37%

### And finally:

### Build:
Added only strides = 2 to second Conv2D.

### Effectiveness on evaluation data:
### App. 70.15% and 71.06% (with smaller batch_size = 32)

12. Testing LSTM's layers and different approaches:

- model from point 11 + changed filters in Conv2D[16, 6x6; 3x3] --> App. 59%
- model from point 11 + changed filters in Conv2D[smaller - both 3x3] --> App. 56%
- model from point 11 + added LSTM's layers (double) --> 50.30%
- model from point 11 + added LSTM's layers without Conv2D --> 57.27%
- model from point 11 + double Conv2D with double LSTM's --> 42.42%
- model from point 11 + double Conv2D with double LSTM (no pool - LSTM after Conv2D) --> 62.88%
- model from point 11 + double Conv2D with double LSTM (instead of pools - LSTM between Conv2D) --> 54.24 %

Tried also adding BatchNormalization and different combinations of Dense layers.

13. Final build - one bidirectional LSTM added with 4 Dense layers.

### Build:
1. Conv2D [32, 6x6]                                                                     
2. Max_pooling2D [2x2]
3. Conv2D [16, 3x3] [strides=2]                                                                     
4. Max_pooling2D [2x2]                                                              
5. BatchNormalization                                                                 
6. Reshape [1x480]
7. Bidirectional LSTM - 320
8. Activation - ReLU
9. BatchNormalization
10. Dense - 320
11. Dropout - 0.5
12. Activation - ReLU 
13. BatchNormalization 
14. Dense - 170 
15. Dropout - 0.5 
16. Activation - ReLU 
17. BatchNormalization 
18. Dense - 170
19. Dropout - 0.5 
20. Activation - ReLU 
21. BatchNormalization 
22. Dense - 65 
23. Dropout - 0.5 
24. Activation - ReLU 
25. BatchNormalization 
26. Dense - 10 
27. Activation - Softmax [10]

Model worked on parameters:
win_length = 256
n_batch_size = 64
n_epoch = 20
n_mels_bank = 20

loss = categorical crossentropy
optimizer = ADAM
metric = accuracy
test_percentage = 0.2

### Effectiveness on evaluation data:
### App. 73.6% and 72.58% (without one dense layer and smaller batch_size = 32)
