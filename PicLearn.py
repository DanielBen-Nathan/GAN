import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D,LeakyReLU,BatchNormalization,Conv2DTranspose,Reshape,UpSampling2D
from tensorflow.keras.optimizers import RMSprop
from keras import backend as K
import pickle
import cv2
import os
from contextlib import redirect_stdout


class Gan:
        def __init__(self,folder,loadFile):
                
                self.folder=folder
                
                self.modelFolder='models'
                self.loadModelNumb=-1
                self.modelName='model'
                self.loadModel=False

                self.noiseNo=100

                self.loadFile=loadFile
                self.picFolder='pics'
                self.picName='pic%d.png'

                self.x=pickle.load(open(loadFile+'.pickle','rb'))

                self.x=self.x/255.0#normalise
                self.imgDim=self.x.shape[1]#width and height
                self.imgDepth=self.x.shape[3]#depth 1 or black and white, 3 for colour
               
                self.D = Sequential()
                self.G = Sequential()
                self.AM = Sequential()

                
                

                

        def descriminator(self,depth,dropout,alpha,window,stridedLayers,unstridedLayers):
                
                #dropout ignore units during training prevent over fitting
                #alpha leaked negative gradient size
                strides=2
                
                input_shape = (self.imgDim, self.imgDim, self.imgDepth)
                self.D.add(Conv2D(depth*1, window, strides=strides, input_shape=input_shape,padding='same'))
                self.D.add(LeakyReLU(alpha=alpha))
                self.D.add(Dropout(dropout))
                

                for i in range(stridedLayers-1+unstridedLayers):
                 
                        depth=depth*2
                        self.D.add(Conv2D(depth, window, strides=strides, padding='same'))
                        self.D.add(LeakyReLU(alpha=alpha))
                        self.D.add(Dropout(dropout))
                     
                        
                        if(i+2==stridedLayers):
                                
                                strides-=1

           


                self.D.add(Flatten())
                self.D.add(Dense(1))#real or not
                self.D.add(Activation('sigmoid'))

                
                

        def gen(self,depth,dim,momentum,dropout,window,upSampledLayers,additionalLayers):

                
               
                self.G.add(Dense(dim*dim*depth, input_dim=self.noiseNo))
                self.G.add(BatchNormalization(momentum=momentum))#stablise learing reduce weight shifting in hidden by reducing by mean
                self.G.add(Activation('relu'))
                self.G.add(Reshape((dim, dim, depth)))#7x7x256
                self.G.add(Dropout(dropout))

                for i in range(upSampledLayers+additionalLayers):
                        
                        
                        depth=int(depth/2)
                        if(i<upSampledLayers):
                                self.G.add(UpSampling2D())#repeats rows and columns e.g 32x32 becomes 64x64
                        self.G.add(Conv2DTranspose(depth, window, padding='same'))#reduce depth to 128
                        self.G.add(BatchNormalization(momentum=momentum))
                        self.G.add(Activation('relu'))

               
            
                self.G.add(Conv2DTranspose(self.imgDepth, window, padding='same'))
                self.G.add(Activation('sigmoid'))
                
               


        def combine(self,dlr,dcv,ddcy,alr,acv,adcy):
              
               
                optimizer = RMSprop(lr=dlr, clipvalue=dcv, decay=ddcy)
                
                
                self.D.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                
                self.D.trainable=False#freeze weights in descriminator so gen weights update instead

                optimizer = RMSprop(lr=alr, clipvalue=acv, decay=adcy)

               
                self.AM.add(self.G)#gen then input to discrim
                self.AM.add(self.D)
                
                
                          
                self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                self.createDir()
                self.writeParams()
               


        def saveImage(self,noise, numb):
               
                filename = self.folder+'/'+self.picFolder+'/'+self.picName % numb
                images = self.G.predict(noise)
                image=images[0]
                

                plt.figure(figsize=(10,10))
                if(self.imgDepth==3):
                    image = np.reshape(image, [self.imgDim , self.imgDim,self.imgDepth])
                    plt.imshow(image)
                else:
                    image = np.reshape(image, [self.imgDim , self.imgDim])
                    plt.imshow(image, cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                
                plt.savefig(filename)
                plt.close('all')
                
        
                        
                
        def loadModel(self,loadModelNumb):
                try:
                        self.loadModel=True
                        self.D=self.load_model(modelLoc(loadModelNumb,'descriminator'))
                        self.G=self.load_model(modelLoc(loadModelNumb,'generator'))
                        self.AM=self.load_model(modelLoc(loadModelNumb,'Adversarial'))
                         
                                
                except OSError as er:
                        print('Error loading file ',er)
                self.createDir()
                self.writeParams()

        def train(self,batchSize,trainingSteps,saveImgInterval,saveModelInterval):
                self.batchSize=batchSize
                self.trainingSteps=trainingSteps
                self.saveImgInterval=saveImgInterval
                self.saveModelInterval=saveModelInterval
                noise = np.random.uniform(-1.0, 1.0, size=[self.batchSize, self.noiseNo])#random noise of 100 units for each batch
                for i in range(self.trainingSteps):
                        
                        
                        if(self.loadModel):
                                i+=self.loadModelNumb+1
                        
                         
                           
                        images_train = self.x[np.random.randint(0,self.x.shape[0],size=self.batchSize),:,:,:]#index random images of batch size remove rand for 1 img
                        noise = np.random.uniform(-1.0, 1.0, size=[self.batchSize, self.noiseNo])#random noise of 100 units for each batch
                        images_fake = self.G.predict(noise)#gen fake image
                           
                        xs = np.concatenate((images_train, images_fake))#concat real then fake
                        y = np.ones([2*self.batchSize, 1])#first half 1 for real second 0 for fake
                        y[self.batchSize:, :] = 0
                          
                        dLoss = self.D.train_on_batch(xs, y)#train on half real half fake images

                           
                        y = np.ones([self.batchSize, 1])#all 1 so mislabled for real
                        noise = np.random.uniform(-1.0, 1.0, size=[self.batchSize, self.noiseNo])#new noise as learned from old noise knows it's wrong, might help
                        aLoss = self.AM.train_on_batch(noise, y)
                        msg = '%d: [D loss: %f, acc: %f]' % (i, dLoss[0], dLoss[1])
                        msg = '%s  [A loss: %f, acc: %f]' % (msg, aLoss[0], aLoss[1])
                        print(msg)
                        if (i % self.saveImgInterval==0):
                                self.saveImage(noise, i)
                        if (i % self.saveModelInterval==0 and i !=0):
                                
                                self.D.save(self.modelLoc(i,'descriminator'))
                                self.G.save(self.modelLoc(i,'generator'))
                                self.AM.save(self.modelLoc(i,'Adversarial'))

                self.D.save(self.modelLoc(self.trainingSteps,'descriminator'))
                self.G.save(self.modelLoc(self.trainingSteps,'generator'))
                self.AM.save(self.modelLoc(self.trainingSteps,'Adversarial'))

                                
        def modelLoc(self,num,netType):
                return self.folder+'/'+self.modelFolder+'/'+str(num)+self.modelName+'_'+netType+'.model'

        def writeParams(self):
                with open(self.folder+'/params.txt', 'w+') as f:
                        size=str(self.imgDim)+'X'+str(self.imgDim)+'X'+str(self.imgDepth)
                        #batchSteps='batch size: '+str(self.batchSize)+' '+'trainingSteps: '+str(self.trainingSteps)
                        f.write(self.folder+'\n'+'\n'+size+'\n')
                        
                        
                        with redirect_stdout(f):
                                self.D.summary()
                        with redirect_stdout(f):
                                self.G.summary()
                        
        def createDir(self):
                if not os.path.exists(self.folder):
                         os.makedirs(self.folder)
                if not os.path.exists(self.folder+'/'+self.picFolder):
                         os.makedirs(self.folder+'/'+self.picFolder)

                if not os.path.exists(self.folder+'/'+self.modelFolder):
                         os.makedirs(self.folder+'/'+self.modelFolder)
                         
if __name__ == '__main__':
        gan=Gan('gnome64','input/gnome64')
        gan.gen(1024, int(gan.imgDim/8),0.9,0.4,5,3,1)
        gan.descriminator(int(gan.imgDim/2),0.4,0.2,5,4,1)
        gan.combine(0.0008,1.0,6e-8,0.0004,1.0,3e-8)
        gan.train(8,1000,1,25)

        


           
