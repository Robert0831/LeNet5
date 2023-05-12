import tensorflow as tf
import numpy as np
from dataset import Dataset50Loader
import os
import time
start=time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation='sigmoid', input_shape=(64,64,3)),
    tf.keras.layers.MaxPool2D(strides=2), #60->30
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='sigmoid', input_shape=(30,30,6)),
    tf.keras.layers.MaxPool2D(strides=2), #30->13
    tf.keras.layers.Conv2D(filters=120, kernel_size=(5,5), padding='valid', input_shape=(13,13,16)),#13->9
    tf.keras.layers.Flatten(),#-1,120*9*9

    tf.keras.layers.Dense( 84, activation = 'sigmoid'),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Softmax(axis=-1),
    ]
)
    
# model.build()
# model.summary()

optimi=tf.optimizers.Adam(learning_rate=0.0001)
train_loader=Dataset50Loader('train2.txt')
val_loader=Dataset50Loader('val.txt')
totalt=len(train_loader.dataset)
totalv=len(val_loader.dataset)
#loaded_model = tf.keras.saving.load_model("model.keras")

@tf.function
def model_func(x,y):
    y_pr=model(x)
    los=tf.keras.losses.SparseCategoricalCrossentropy()(y,y_pr)
    return y_pr, los

for epoch in range(50):
    act=0
    losst=[]
    for batch_idx, (img,data) in enumerate(train_loader):
        img=np.transpose(img.numpy(),(0,2,3,1)).astype(np.float32)
        data=data.numpy().astype(np.float32)

        with tf.GradientTape() as tape:
            # y_pred=model(img)
            # los=tf.keras.losses.SparseCategoricalCrossentropy()(data,y_pred)
            y_pred,los=model_func(img,data)
            losst.append(tf.get_static_value(los))
            grad=tape.gradient(los,model.trainable_variables)
            optimi.apply_gradients(zip(grad,model.trainable_variables))

        y_pred=np.argmax(y_pred,axis=1)
        for i in range(len(data)):
            if y_pred[i]==data[i]:
                act+=1

        if batch_idx%50==0:
            print(f'epoch={epoch} ,batch={batch_idx},loss={los}')   


    logger = open('tf_sta_train.txt', 'a')
    logger.write('%d %f %f\n'%(epoch,round(sum(losst)/len(losst),3),round(act/totalt,3)))
    logger.close()   

    acv=0
    lossv=[]
    for batch_idx, (img,data) in enumerate(val_loader):
        img=np.transpose(img.numpy(),(0,2,3,1)).astype(np.float32)
        data=data.numpy().astype(np.float32)

        
        y_pred=model(img)
        los=tf.keras.losses.SparseCategoricalCrossentropy()(data,y_pred)
        lossv.append(tf.get_static_value(los))

        y_pred=np.argmax(y_pred,axis=1)
        for i in range(len(data)):
            if y_pred[i]==data[i]:
                acv+=1

    logger = open('tf__sta_val.txt', 'a')
    logger.write('%d %f %f\n'%(epoch,round(sum(lossv)/len(lossv),3),round(acv/totalv,3)))
    logger.close()  
model.save("tf_model_static.keras" )
print(time.time() - start)