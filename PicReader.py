import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

class PictureReader:
    def __init__(self,path,save,imgSize,picNo,colour,show):
        self.path=path
        self.save=save
        self.IMG_SIZE=imgSize
        self.PICNO=picNo
        self.colour=colour
        self.show=show

        self.training_data=[]
        self.saveLoad()


    def create_training_data(self):
       
        
        count=0
        for img in os.listdir(self.path):
            
            try:
                img_array=[]
                if (self.colour):
                    
                    imgArray=cv2.imread(os.path.join(self.path,img))
                    imgArray = cv2.cvtColor(imgArray, cv2.COLOR_BGR2RGB)
                    imgArray=cv2.resize(imgArray,(self.IMG_SIZE,self.IMG_SIZE))
                    plt.imshow(imgArray)
                else:
                    
                    imgArray=cv2.imread(os.path.join(self.path,img),cv2.IMREAD_GRAYSCALE)
                    imgArray=cv2.resize(imgArray,(self.IMG_SIZE,self.IMG_SIZE)).astype(np.float32)
                    plt.imshow(imgArray,cmap='gray')
                
               
                if (self.show):
                    plt.show()
                    
                self.training_data.append(imgArray)#data and label
                count+=1
            except Exception as e:
                
                pass
            if(self.PICNO<=count):
                break;

    def format_data(self):
        x=[]
       
        for features in self.training_data:
            x.append(features)
            
        if(self.colour):
            depth=3
        else:
            depth=1
        x=np.array(x).reshape(-1,self.IMG_SIZE,self.IMG_SIZE,depth).astype(np.float32)#would be 3 for colour instead 1, -1for anything
        return x


    def saveLoad(self):
        if (not os.path.isfile(self.save+'.pickle')):
            print('setup data')
            self.create_training_data()
            random.shuffle(self.training_data)
            x=self.format_data()
            


            pickle_out=open(self.save+'.pickle','wb')
            pickle.dump(x,pickle_out)
            pickle_out.close()

           
        else:
            print('file already exists')
            pickle_in=open(self.save+'.pickle','rb')
            x=pickle.load(pickle_in)
        
        print("Finished")
                           
if __name__ == '__main__':         
    #reader=PictureReader('data/face','input/face64',64,1000,True,False)
    reader=PictureReader('data/gnome','input/gnome48',48,1,True,True)

                          

