from os import listdir
from os.path import isfile,isdir, join
import numpy
import datetime
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop
ih, iw = 150, 150 #tamano de la imagen
input_shape = (ih, iw, 3) #forma de la imagen: alto ancho y numero de canales
#train_dir = 'data/minitrain' #directorio de entrenamiento
#test_dir = 'data/minitest' #directorio de prueba
train_dir = 'data/train' #directorio de entrenamiento
test_dir = 'data/test' #directorio de prueba

num_class = 2 #cuantas clases
epochs = 30 #cuantas veces entrenar. En cada epoch hace una mejora en los parametros
batch_size = 50 #batch para hacer cada entrenamiento. Lee 50 'batch_size' imagenes antes de actualizar los parametros. Las carga a memoria
num_train = 1200 #numero de imagenes en train
num_test = 1000 #numero de imagenes en test

epoch_steps = num_train // batch_size
test_steps = num_test // batch_size

gentrain = ImageDataGenerator(rescale=1. / 255.) #indica que reescale cada canal con valor entre 0 y 1.

train = gentrain.flow_from_directory(train_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')
gentest = ImageDataGenerator(rescale=1. / 255)
test = gentest.flow_from_directory(test_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')
#para cargar pesos de la red desde donde se quedó la ultima vez
#filename = "cvsd.h5"
#model.load_weights(filename)  #comentar si se comienza desde cero.
###
model = Sequential()
model.add(Conv2D(10, (3, 3), input_shape=(ih, iw,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(10, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
python -m tensorboard.main --logdir=/Graph  <- Para correr Tensor board
#tensorboard  --logdir Graph/
print("Logs:")
print(log_dir)
print("__________")

model.fit_generator(
                train,
                steps_per_epoch=epoch_steps,
                epochs=epochs,
                validation_data=test,
                validation_steps=test_steps,
                callbacks=[tbCallBack]
                )

path="data/test"
carpetas = [ f for f in listdir(path) if isdir(join(path,f))]
import re
#Ordenar Carpetas
_nsre = re.compile('([0-9]+)')
def nort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]
carpetas.sort()
predclass=[None] *len(carpetas)
realclass=[None] *len(carpetas)
for n in range(0, len(carpetas)) :
        numeroclasses=[]
        prediccion=[]
        print(carpetas[n])
        claseactual = [ f for f in listdir(join(path,carpetas[n])) if isfile(join(path,carpetas[n],f))]
        for m in range(0, len(claseactual)) :
                imagen = image.load_img(join(path,carpetas[n],claseactual[m]), target_size=(iw, ih))
                imagen = image.img_to_array(imagen)
                imagen /= 255
                imagen = numpy.expand_dims(imagen, axis=0)
                #deteccion=(model.predict_classes(imagen,verbose='0') )
                deteccion=(model.predict(imagen) > 0.5).astype("int32")
                prediccion.append(deteccion)
                numeroclasses.append(n)
        predclass[n]=prediccion
        realclass[n]=numeroclasses

        print(sum(predclass[n]))
        print(sum(realclass[n]))
s0=(2500-sum(predclass[0]))
s1=(sum(predclass[1]))
a0=(s0+s1)/float(5000)
print(train.class_indices)
print("Aciertos")
print(a0)
model.save('cvsd.h5')