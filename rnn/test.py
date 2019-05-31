import numpy as np
from time import time
from keras.models import Sequential
from keras.layers import GRU,Bidirectional,Dense,Dropout,AveragePooling2D,Flatten
import matplotlib.pyplot as plt 


data_fixed_length = 100
number_of_classes = 50
epochs = 10
batch_size = 200

class Data_prporcessing:

    def execute(self):
        self.loadInternalRepresentationFiles()
        self.augumentDataSets()
        self.toNpArrs()
        return self.train_set, self.train_labels, self.test_set, self.test_labels

    def toNpArrs(self):
        def toNpArr(s):
            new = []
            for i in s:
                temp = []
                for j in i:
                    temp.append(np.asarray(j))
                new.append(np.asarray(temp))
            return np.array(new)
        self.train_set = toNpArr(self.train_set)
        self.test_set = toNpArr(self.test_set)

    def augumentDataSets(self):
        def augumentDataSet(dataset):
            for i in range(len(dataset)):
                if len(dataset[i]) > data_fixed_length:
                    dataset[i] = dataset[i][:data_fixed_length]
                else:
                    dataset[i] = dataset[i] + [[0] * 6] * (data_fixed_length - len(dataset[i]))
        augumentDataSet(self.train_set)
        augumentDataSet(self.test_set)


    def loadInternalRepresentationFiles(self):
        start = time()
        print('reading representation files...')
        self.train_set = np.load("trainset.npy")
        self.train_labels = np.load('trainlabels.npy')
        self.test_set = np.load('testset.npy')
        self.test_labels = np.load('testlabel.npy')
        print("4 np files read in",time() - start,"seconds")
        self.convertLabelsToKeys()
        self.train_labels = np.reshape(np.array(self.train_labels),(len(self.train_labels),1))
        self.test_labels = np.reshape(np.array(self.test_labels),(len(self.test_labels),1))
        print("representation files have been loaded")


    def convertLabelsToKeys(self):
        def defineDict(labels):
            label_dic = {}
            current_index = 0
            for l in labels:
                if l not in label_dic.keys():
                    label_dic[l] = current_index
                    current_index += 1
            return label_dic,{v: k for k, v in label_dic.items()}
        self.l2k,self.k2l = defineDict(self.test_labels)
        def toClassArray(labels):
            new_labels = [[]]
            for l in labels:
                new_i = [0] * number_of_classes
                new_i[l] = 1
                new_labels.append(new_i)
            return new_labels[1:]

        self.train_labels = [self.l2k[i] for i in self.train_labels]
        self.test_labels = [self.l2k[i] for i in self.test_labels]


class rnn:
    def buildRNN(self):
        print('Building model')
        model = Sequential()
        model.add(Bidirectional(GRU(500, return_sequences=True), merge_mode='sum'))
        # model.add(Bidirectional(GRU(300, return_sequences=True),merge_mode='sum'))
        model.add(Flatten())
        model.add(Dense(number_of_classes, activation='softmax'))
        print('compiling model')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
        return model

    def shuffle(self):
        pass
    
    def plotHistory(self):
        # list all data in history

        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()






data = Data_prporcessing()
train_set, train_labels, test_set, test_labels = data.execute()



# rnn = rnn()
# model = rnn.buildRNN()
# start = time()
# history = model.fit(data.train_set, data.train_labels,
#                     validation_split=0.3,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     shuffle=True,
#                     verbose= 1)
# print("time cost :", time()-start, "seconds")

# rnn.plotHistory()




