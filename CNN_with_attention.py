'''
Custom CNN with attention
Here I've used a CNN architecture of my own for the images and trained it from scratch, as opposed to using a pre-trained VGG network.
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
from keras.layers import Dense, Embedding, LSTM, Input,Dropout,Merge,Lambda, Conv2D,RepeatVector,Reshape,Multiply, MaxPooling2D
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
    
tokenizer = Tokenizer(num_words=1000, lower=True,split=' ')
tokenizer.fit_on_texts(ques_lis)
questions = tokenizer.texts_to_sequences(ques_lis)
word_index = tokenizer.word_index


max_length_of_text = 25
questions = pad_sequences(questions, maxlen = max_length_of_text)

#print data shapes
print('questions: ' + str(len(questions))  + '   ' + str(questions[0].shape)  )
print('answers: ' + str(len(answers)) + '   '  + str(answers[0].shape) )
print('images: ' + str(len(img_lis)) + '   '  )
   
   
#make lstm model
ques_input = Input((max_length_of_text, ))
x = Embedding(input_dim = len(word_index) + 1 , output_dim=256, input_length = max_length_of_text)(ques_input)
x = LSTM(units=256, return_sequences=True)(x)
x = Dropout(0.1)(x)
x = LSTM(units = 512, return_sequences=False)(x)
x = Dropout(0.1)(x)
question_tensor = Dense(256, activation='tanh')(x)
    

#make vgg model


image_input = Input(shape=(120, 160, 3))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(image_input)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(196, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(196, (3, 3), activation='relu')(x)
x = Conv2D(196, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
vgg_out = MaxPooling2D((2, 2))(x)                             # (4,6,256) is the output dim


def attention(tensors):
    img = tensors[0]    # (4,6,256)
    ques = tensors[1]   # (256,)
    print('attention: img shape: '+ str( K.int_shape(img) ) )
    print('attention: ques shape: '+ str( K.int_shape(ques) ) )
   
    ques = RepeatVector(24)(ques)
    ques = Reshape((4, 6, -1))(ques)
   
    print('attention: ques shape 2: '+ str( K.int_shape(ques) ) )
    
    x = Conv2D(64,(1,1),activation = 'relu')(img)    
    y = Conv2D(64,(1,1),activation = 'relu')(ques)

    z = tf.multiply(x,y)
    print('attention: multiply shape : '+ str( K.int_shape(z) ) )
    z = Conv2D(1,(1,1),activation='linear')(z)
    z = tf.exp(z)
    den = tf.reduce_sum(z,axis=0)
    z = tf.div(z,den)
    
    img2 = tf.multiply(img,z)
    img2 = tf.reduce_sum( tf.reduce_sum(img2 ,axis=1)  , axis=1  )
    print('attention: img shape after reduce sum: '+ str( K.int_shape(img2)  ) )
    return img2

def attention_shape(tensors):
    print('attention shape:' + str(tensors))
    return tensors[1]

layer = Lambda(attention, attention_shape)

img_tensor = layer( [vgg_out, question_tensor] )     # (256,)

#make final model

y = Multiply()([img_tensor, question_tensor])
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

#--
img_lis_train = img_lis[ : 108032]
img_lis_test  = img_lis[ 108032 : ]
questions_train = questions[ : 108032]
questions_test  = questions[ 108032 : ]
answers_train = answers[ : 108032]
answers_test  = answers[ 108032 : ]


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
            img = load_img('./dataset/images/' + str(img_name) + '.png' , target_size = (120,160))
            img = img_to_array(img)
            img = img/255.0   
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
        #self.val_losses = []
        
        self.acc = []
        #self.val_acc = []
       
        self.logs = []

    def on_batch_end(self, batch, logs={}):
    
        if self.batch % self.N == 0:
            #score = fc_model.evaluate_generator(mygen_val(questions,img_lis,answers)  ,steps=420)  
            
            self.logs.append(logs)
            self.x.append(self.batch)
            self.losses.append(logs.get('loss'))
            #self.val_losses.append(  score[0]  )
            self.acc.append(logs.get('acc'))
            #self.val_acc.append( score[1]  )
        
            print()
            print('Train Acc: ' + str(  logs.get('acc')   ))
            print('Train loss: ' + str(  logs.get('loss')   ))
            #print('Test Acc: ' + str( score[1] ))
            print()
        
            
            #clear_output(wait=True)
            f = plt.figure(1)
            plt.plot(self.x, self.losses, label="loss")
            #plt.plot(self.x, self.val_losses, label="val_loss")
            plt.title('Training loss vs batches')
            plt.xlabel('batches')
            plt.ylabel('loss')
            f.savefig('./plots/loss_plot.jpg')
            
            g = plt.figure(2)
            plt.plot(self.x, self.acc, label="acc")
            #plt.plot(self.x, self.val_acc, label="val_acc")
            plt.title('Training acc vs batches')
            plt.xlabel('batches')
            plt.ylabel('accuracy')
            g.savefig('./plots/acc_plot.jpg')
            
            
        self.batch += 1
      


class WeightsSaver(Callback): 
    def __init__(self, N):
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = './weights/weights.h5'
            self.model.save_weights(name)
        self.batch += 1

#fc_model.load_weights('./weights/weights.h5')
fc_model.fit_generator(mygen(questions_train, img_lis_train , answers_train ), steps_per_epoch = 1688, epochs = 100 ,callbacks=[WeightsSaver(60), TestCallback(400)])


