import network 
import pickle 
import mnist_loader 


DTraining, DValidation, DTest = mnist_loader.load_data_wrapper()

DTraining = list(DTraining)
DTest = list(DTest)

net = network.Network([784, 30, 10])
net.SGD(DTraining, 30, 10, 3.0, test_data = DTest)

File = open('red_prueba.pkl','wb')
pickle.dump(net, File)
File.close()
exit()

Leer_archivo = open('red_prueba.pkl','rb')
net = pickle.load(Leer_archivo)
Leer_archivo.close()

net.SGD(DTraining, 10, 50, 3.0, test_data = DTest)

File = open('red_prueba.pkl', 'wb')
pickle.dump(net, File)
File.close()