# Guided CTC architecture for text recognition
 The architecture has beeen proposed for text recognition, which exploit both attention and CTC. These two architecture are mainstream for text recognition.
 As a supervised approach, inputs contain (image, label)
 * train Dataset: SynthText , MJsynth
 * test Dataset : IC03, IC13, IC15, IIIT5K , SVT 
   * Input Images: text images wth varying size.
   * Labels : text(concatenation of characters)[transcription of the text image]
   * as we said, we are combining attention and CTC. as the label consideration for each of these method is different, we need some modification on GT labels
   * after encoding character to index, for:
   * attention : we add SoS(start of sequence) and EoS(end of sequence) character for each text sequence.
   * CTC: is based on repetition and blank label. 
   * anyway, both CTC and attention code, need same length of text in each batch which is identified based on maximum length of text in each batch and padding the remaining based on this maximum length in that batch   
   

    
   
 
 The architecture is composed of four main submoduls:
 ## 1.  STN
 If the image is not horizontally alligned, it is needed to transform input image to achieve normalized image.
 
 ### 1) localization network
 to predict transformation parameters
 ### 2) grid generator
 * in our model, rectification is done manually throught the manual transformation.
 * As another normalization step, text images are resized to fixed height
 * we will have fixed height and variable width images. 
 As batches as input to model, should be in the same size. width of images per batches are padded to the maximum width of the image in that batches
 * 
 ## 2. Feature Extraction
 As the most common architecture for extracting features from images, CNN is applied
 based in computational power, Resnet or light mobilenet.
 Size : (Batchsize, w , h, c) 
 ### 1) ResNet
  ResNet50 is applied with some modification
 ### 1)* Mobilenet
* extracted feature maps should be transformed to feature vectors
* Size(Batchsize, w , h*c)
* So far we have encoded the images, next step is decoding.

we should apply decoder as input for decoder parts for training.

Attention Decoder need some weights for decoder inputs, too. we consider decoder input with a constant length(maximum length text in all texes+2[as SOS and EOS]).

CTC decoder input is just the text encoded array.

Note: based on the paper the first three submodules(STN, ResNet-CNN and the attentional guidance) are solely trained with cross entropy loss.while the
GCN+CTC decoder is trained with CTC loss.
 ## 3. Attentional guidance
 the attention decoder 

 ## 4. Graph Convolutional Network(GCN)powered CTC decoder
 the CTC decoder 
##############################
*) config file is considered as the configuration file for model

*) a generator is used to create batches of data, for model fitting

*) images should be resized by scale to: [64,None]


1) first install the require packages in requirement.txt
2) train.sh for training the network
3) test.sh for test 
4) tensorboard.sh for tensorboard visualization


####################

# Transformer Section:

data_generator.py : you can choose which dataset and which mode(transformer for ctc)
 
train : train_transformer_final.py
some related files are available in Models (Layer,Model, callbacks, loss,metric,etc)



   
