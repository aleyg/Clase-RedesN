import CrossEntropy 
import pickle 
import mnist_loader 


Training, Validation, Test = mnist_loader.load_data_wrapper()

Training = list(Training)
Test = list(Test)

net = CrossEntropy.Network([784, 30, 10],cost=CrossEntropy.CrossEntropyCost)
net.SGD(Training, 30, 10, 0.5,0.0, test_data = Test)

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