import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

imgDim=32
model=tf.keras.models.load_model('fake32x32\\Genmnist1.model')
noise=[]
noise = np.random.uniform(-1.0, 1.0, size=[1, 100])
#for i in range(0,100):
    #noise.append(np.random.uniform(-1.0, 1.0))
print(noise)
image=model.predict(noise)[0]

image = np.reshape(image, [imgDim, imgDim])
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
