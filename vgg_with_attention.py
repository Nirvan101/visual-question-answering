'''
VGG with attention
Here I've used a pre-trained VGG network to extract image features. The last few layers are trainable.
I've used the attention mechanism to improve accuracy.
'''

import numpy as np
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import os,cv2
from keras.callbacks import Callback
import json
import tensorflow as tf
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input,Dropout,Merge,Lambda, Conv2D,RepeatVector,Reshape,Multiply, MaxPooling2D, concatenate
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from random import shuffle
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import keras.backend as K


labels  =  {0 : 0,
 1 : 1,
 2 : 2,
 3: 3,
 4: 4,
 5 : 5,
 6 : 6,
 7 : 7,
 8 : 8,
 False : 9,
 True : 10,
 'blue' : 11,
 'brown' : 12,
 'cube': 13,
 'cyan' : 14,
 'cylinder': 15, 
 'gray' : 16,
 'green' : 17,
 'large' : 18,
 'metal' : 19,
 'purple' : 20,
 'red' : 21,
 'rubber': 22,
 'small' : 23,
 'sphere' : 24,
 'yellow' : 25  }

path = './dataset/Quest_Answers.json'

with open(path) as json_data:
    dic = json.load(json_data)
    
dic = dic.get('quest_answers')   
    
ques_lis = []
img_lis = []
ans_lis = []
    
for el in dic:  
    ques_lis.append( el.get('Question') )
    ans_lis.append( el.get('Answer') )
    img_name = el.get('Image')
    img_lis.append(img_name)
  
answers = [labels.get(l) for l in ans_lis]  
answers = to_categorical(answers, num_classes=26)
    
## MAKE THE MODEL  ---------------------------------------------------------------------------------------------------
    
tokenizer = Tokenizer(num_words=1000, lower=True, split=' ')
tokenizer.fit_on_texts(ques_lis)
questions = tokenizer.texts_to_sequences(ques_lis)
word_index = tokenizer.word_index


max_length_of_text = 25
questions = pad_sequences(questions, maxlen = max_length_of_text)

   
#make lstm model
ques_input = Input((max_length_of_text, ))
x = Embedding(input_dim = len(word_index) + 1 , output_dim=256, input_length = max_length_of_text)(ques_input)
x = LSTM(units=256, return_sequences=True)(x)
x = Dropout(0.1)(x)
x = LSTM(units = 512, return_sequences=False)(x)
x = Dropout(0.1)(x)
question_tensor = Dense(512, activation='tanh')(x)
    

#make vgg model
vgg = VGG16(weights=None, include_top=True)
vgg.load_weights('./vgg_weights.h5')
vgg.layers.pop()
vgg.layers.pop()
vgg.layers.pop()
vgg.layers.pop()
#output is (7,7,512)


for l in vgg.layers[:-6]:
    l.trainable = False
for l in vgg.layers[-6:]:
    l.trainable = True

    
image_input = vgg.input
vgg_out = vgg.layers[-1].output     # (7,7,512)



def attention(tensors):
    img = tensors[0]    # (7,7,512)
    ques = tensors[1]   # (512,)
    print('attention: img shape: '+ str( K.int_shape(img) ) )
    print('attention: ques shape: '+ str( K.int_shape(ques) ) )
   
    ques = RepeatVector(49)(ques)
    ques = Reshape((7, 7, -1))(ques)           # (7,7,512)
   
    print('attention: ques shape 2: '+ str( K.int_shape(ques) ) )
    
    x = Conv2D(64,(1,1),activation = 'relu')(img)      # (7,7,64)    
    y = Conv2D(64,(1,1),activation = 'relu')(ques)     # (7,7,64)

    z = tf.multiply(x,y)
    print('attention: multiply shape : '+ str( K.int_shape(z) ) )
    z = Conv2D(1,(1,1),activation='linear')(z)          # (7,7,1)
    z = tf.exp(z)
    den = tf.reduce_sum(z)
    z = tf.div(z,den)
    
    img2 = tf.multiply(img,z)
    img2 = tf.reduce_sum( tf.reduce_sum(img2 ,axis=1)  , axis=1  )
    print('attention: img shape after reduce sum: '+ str( K.int_shape(img2)  ) )
    return img2    # (1,1,512)

def attention_shape(tensors):
    print('attention shape:' + str(tensors))
    return tensors[1]

layer = Lambda(attention, attention_shape)     

img_tensor = layer( [vgg_out, question_tensor] )     # (512,)

#make final model
y = concatenate([img_tensor, question_tensor])   #(1024,)
y = Dropout(0.2)(y)
y = Dense(256, activation='relu')(y)
y = Dropout(0.2)(y)
y = Dense(26, activation='softmax')(y)
fc_model = Model( [image_input,ques_input] , y)
fc_model.compile(optimizer='adam', loss='categorical_crossentropy',
    metrics=['accuracy'])

fc_model.summary()


'''
if os.path.exists(model_weights_filename):
   print "Loading Weights..."
   fc_model.load_weights(model_weights_filename)
       
'''

 
#Train


img_lis_train = img_lis[ : 132992]
img_lis_test  = img_lis[ -1984 : ]
questions_train = questions[ : 132992]
questions_test  = questions[ -1984 : ]
answers_train = answers[ : 132992]
answers_test  = answers[ -1984 : ]

def mygen(questions_train,img_lis_train,answers_train):
    start = 0  
    data_size = len(questions_train)
    batch_size = 64
    while True:          
        if( start+batch_size <= data_size ):
            batch_ques = questions_train[ start : start+batch_size ] 
            batch_ans = answers_train[ start : start+batch_size ] 
            batch_img_names = img_lis_train[ start : start+batch_size ] 
        elif(start < data_size):
            batch_ques = questions_train[ start : ] 
            batch_ans = answers_train[ start : ] 
            batch_img_names = img_lis_train[ start : ] 
        else:
            start = 0
            continue     
  
        batch_img = []
        for img_name in batch_img_names:
            img = load_img('./dataset/images/' + str(img_name) + '.png' , target_size = (224,224))
            img = img_to_array(img)
            img = preprocess_input(img)  
            batch_img.append( img )    
                
        start += batch_size
        print('start = ' + str(start))
        yield [np.array(batch_img), np.array(batch_ques)] ,np.array(batch_ans)


class TestCallback(Callback):

    def __init__(self,N):
        self.N = N
        self.batch = 0
    
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.acc = []
        self.val_acc = []
       
        self.logs = []

    def on_batch_end(self, batch, logs={}):    
        if self.batch >= 6000:
            self.batch += 1
            return
            
        if self.batch % self.N == 0:
            #score = fc_model.evaluate_generator(mygen_val(questions,img_lis,answers)  ,steps=420)  
            
            self.logs.append(logs)
            self.x.append(self.batch)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(  logs.get('val_loss'))
            self.acc.append(logs.get('acc'))
            self.val_acc.append( logs.get('val_acc')  )
        
            
            #clear_output(wait=True)
            f = plt.figure(1)
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            plt.title('Training loss vs batches')
            plt.xlabel('batches')
            plt.ylabel('loss')
            f.savefig('./plots/vgg_with_attention_loss_plot.jpg')
            
            g = plt.figure(2)
            plt.plot(self.x, self.acc, label="acc")
            plt.plot(self.x, self.val_acc, label="val_acc")
            plt.title('Training acc vs batches')
            plt.xlabel('batches')
            plt.ylabel('accuracy')
            g.savefig('./plots/vgg_with_attention_acc_plot.jpg')
            
            
        self.batch += 1
      


class WeightsSaver(Callback): 
    def __init__(self, N):
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = './weights/model_vgg_with_attention_batchwise.h5'
            self.model.save(name)
        self.batch += 1

#fc_model.load_weights('./weights/weights.h5')

train_gen = mygen(questions_train, img_lis_train , answers_train )
val_gen = mygen(questions_test, img_lis_test , answers_test )

filepath='./weights/model_vgg_with_attention_chkpt.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',save_weights_only = False)

fc_model.fit_generator(train_gen, steps_per_epoch = 2078, validation_data = val_gen, validation_steps = 31 ,epochs = 100 ,callbacks=[WeightsSaver(30), TestCallback(400),checkpoint])


