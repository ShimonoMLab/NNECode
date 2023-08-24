import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import Adam
from sklearn.metrics import log_loss
import os
import glob
import tensorflow as tf
import networkx as nx
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

##
os.environ["PYTHONHASHSEED"] = "0"

np.random.seed(1)   
rn.seed(12345)

import sys
numlayer = int(sys.argv[1])
nummiddle = int(sys.argv[2])

dirda = "../../../data/180731_190806_dateset/" + "l" + str(numlayer) + "_m" + str(nummiddle) + "/"
shutil.rmtree(dirda)
os.mkdir(dirda)

from keras import backend as K
tf.random.set_seed(1234)

kf = KFold(n_splits=5,shuffle=False)
data_list = ["180731001","190423001","190514002","190604001","190611001","190617002","190617004","190806002","180731002","190514001","190514003","190604002","190617001","190617003","190806001"]
kf.get_n_splits(data_list)
data_loss_value = pd.DataFrame()
data_loss_value_test = pd.DataFrame()

nfolds = 1
for train_index, test_index in kf.split(data_list):
   list_train0 = np.delete(data_list,test_index)
   list_test = np.delete(data_list,train_index)
   list_valid = list_train0[9:]
   list_train = list_train0[:9]
   data_train = pd.DataFrame()

   for data in list_train:
       directoryda = "../../../../Sharing_SizeSame/{}/pdf.txt".format(data)
       cellcateg  = pd.read_csv("../../../../Sharing_SizeSame/{}/cell_categ_exc.txt".format(data),header = None,sep=",")
       layercateg = pd.read_csv("../../../../Sharing_SizeSame/{}/layer_categ.txt".format(data),header = None,sep=",")
       categ = (cellcateg)*8+(7-layercateg) # inh (deep-->surface)-->exc(deep-->surface)
       data1 = pd.read_csv(directoryda,header=None,sep=",")
       data1["CCat"] = categ[0].values.T
       data1 = data1.sort_values("CCat",ascending=False, kind="mergesort") #****
       data1 = data1.drop("CCat",axis=1)   # 1 --> 0
       data1 = data1.T
       data1["CCat"] = categ[0].values.T
       data1 = data1.sort_values("CCat",ascending=False, kind="mergesort") #****
       data1 = data1.drop("CCat",axis=1)   # 1 --> 0
       data1 = data1.iloc[0:100,0:100]
       data1 = data1.T 
       data1.index = range(100)
       data1.columns = range(100)
       if data_train.empty == True:
           data_train = data1
       else:
           data_train = pd.concat([data_train,data1])

   data_valid = pd.DataFrame()
   for data in list_valid:
       directoryda = "../../../../Sharing_SizeSame/{}/pdf.txt".format(data)
       cellcateg  = pd.read_csv("../../../../Sharing_SizeSame/{}/cell_categ_exc.txt".format(data),header = None,sep=",")
       layercateg = pd.read_csv("../../../../Sharing_SizeSame/{}/layer_categ.txt".format(data),header = None,sep=",")
       categ = (cellcateg)*8+(7-layercateg) # inh (deep-->surface)-->exc(deep-->surface)
       data1 = pd.read_csv(directoryda,header=None,sep=",")
       data1["CCat"] = categ[0].values.T
       data1 = data1.sort_values("CCat",ascending=False, kind="mergesort") #****
       data1 = data1.drop("CCat",axis=1)   # 1 --> 0
       data1 = data1.T
       data1["CCat"] = categ[0].values.T
       data1 = data1.sort_values("CCat",ascending=False, kind="mergesort") #****
       data1 = data1.drop("CCat",axis=1)   # 1 --> 0
       data1 = data1.iloc[0:100,0:100]
       data1 = data1.T 
       data1.index = range(100)
       data1.columns = range(100)
       if data_valid.empty == True:
           data_valid = data1
       else:
           data_valid = pd.concat([data_valid,data1])


   data_test = pd.DataFrame()
   for data in list_test:
       directoryda = "../../../../Sharing_SizeSame/{}/pdf.txt".format(data)
       cellcateg  = pd.read_csv("../../../../Sharing_SizeSame/{}/cell_categ_exc.txt".format(data),header = None,sep=",")
       layercateg = pd.read_csv("../../../../Sharing_SizeSame/{}/layer_categ.txt".format(data),header = None,sep=",")
       categ = (cellcateg)*8+(7-layercateg) # inh (deep-->surface)-->exc(deep-->surface)
       data1 = pd.read_csv(directoryda,header=None,sep=",")
       data1["CCat"] = categ[0].values.T
       data1 = data1.sort_values("CCat",ascending=False, kind="mergesort") #***
       data1 = data1.drop("CCat",axis=1)   # 1 --> 0
       data1 = data1.T
       data1["CCat"] = categ[0].values.T
       data1 = data1.sort_values("CCat",ascending=False)
       data1 = data1.drop("CCat",axis=1)   # 1 --> 0
       data1 = data1.iloc[0:100,0:100]
       data1 = data1.T 
       data1.index = range(100)
       data1.columns = range(100)
       if data_test.empty == True:
           data_test = data1
       else:
           data_test = pd.concat([data_test,data1])

   #data_import
   x_train = data_train
   y_train = data_train.index
   x_test = data_test
   y_test = data_test.index
   x_valid = data_valid
   y_valid = data_valid.index
     
   #log export
   file_history = "./" + "history_folds_" + str(nfolds) + "_" + str(numlayer) + "_" + str(nummiddle) + "_valid.txt"
   callbacks = []
   from keras.callbacks import CSVLogger
   callbacks.append(CSVLogger(file_history))
   from keras.callbacks import EarlyStopping
   #early stopping
   callbacks.append(EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto'))

   random.seed(1)
   autoencoder = Model()
   encoder = Model()
   delnodes = (100-nummiddle)/((numlayer - 1)/2)
   delnode = int(delnodes)
   maxdelnode = 100 - delnode*((numlayer - 1)/2-1)
   maxdelnode
   input_img = Input(shape=(100,))
   print(input_img)
   prevlayers = ['input_img']
   ranklayer = 0
   for i in range(0,numlayer - 2):
       ranklayer = i + 1
       if i < int((numlayer - 2)/2):
           tempnodes = int(100 - delnodes*ranklayer)
           prevlayer = 'encoded' + str(ranklayer)
           prevlayers.append(prevlayer)
           print('encoded' + str(ranklayer) + ' = ' + 'Dense(' + str(tempnodes) + ', activation=\'relu\')' + '(' + prevlayers[i] + ')')
           exec('encoded' + str(ranklayer) + ' = ' + 'Dense(' + str(tempnodes) + ', activation=\'relu\')' + '(' + prevlayers[i] + ')')
       elif i == int((numlayer - 2)/2):
           tempnodes = nummiddle
           prevlayer = 'encoded' + str(ranklayer)
           prevlayers.append(prevlayer)
           print('encoded' + str(ranklayer) + ' = ' + 'Dense(' + str(tempnodes) + ', activation=\'relu\')' + '(' + prevlayers[i] + ')')
           exec('encoded' + str(ranklayer) + ' = ' + 'Dense(' + str(tempnodes) + ', activation=\'relu\')' + '(' + prevlayers[i] + ')')
       elif i > int((numlayer - 2)/2):
           modranklayer = ranklayer - int((numlayer - 2)/2)
           tempnodes = int(nummiddle + delnodes*(modranklayer - 1))
           prevlayer = 'decoded' + str(modranklayer)
           prevlayers.append(prevlayer)
           print('decoded' + str(modranklayer) + ' = ' + 'Dense(' + str(tempnodes) + ', activation=\'relu\')' + '(' + prevlayers[i] + ')')
           exec('decoded' + str(modranklayer) + ' = ' + 'Dense(' + str(tempnodes) + ', activation=\'relu\')' + '(' + prevlayers[i] + ')')
   print(i)
   ranklayer = i + 1 + 1
   modranklayer = ranklayer - int((numlayer - 2)/2) - 1
   tempnodes = 100
   print('decoded' + str(modranklayer) + ' = ' + 'Dense(' + str(tempnodes) + ', activation=\'sigmoid\')' + '(' + prevlayers[(numlayer - 2)] + ')')
   exec('decoded' + str(modranklayer) + ' = ' + 'Dense(' + str(tempnodes) + ', activation=\'sigmoid\')' + '(' + prevlayers[(numlayer - 2)] + ')')

   final_layer = 'decoded' + str(modranklayer)

   exec('autoencoder = Model(inputs=input_img, outputs=' + final_layer +')')
   exec('encoder = Model(inputs=input_img, outputs=' + prevlayers[int((numlayer - 2)/2 + 1)] +')' )
   print(prevlayers[int((numlayer - 2)/2)])
   #optimization
   optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
   autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy",metrics=["binary_accuracy"])
   encoder.compile(optimizer=optimizer, loss="binary_crossentropy",metrics=["binary_accuracy"])
   autoencoder.fit(x_train, x_train,
                   epochs=5000,
                   batch_size=100,
                   shuffle=False,
                   callbacks=callbacks,
                   validation_data = (x_valid,x_valid))

   encoded_vectda = np.array(encoder.predict(x_train))
   encoded_vectda = pd.DataFrame(encoded_vectda)
   encoded_vectda.index = y_train
   file_encode_csv = "./output_middle_layer_" + str(numlayer) + "_" + str(nummiddle) + "_valid.txt"
   encoded_vectda.to_csv(file_encode_csv, sep='\t')

   decoded_vectda = np.array(autoencoder.predict(x_test))
   decoded_vectda = pd.DataFrame(decoded_vectda)
   decoded_vectda.index = y_test
   file_decode_csv = "./output_output_layer_" + str(numlayer) + "_" + str(nummiddle) + "_valid.txt"
   decoded_vectda.to_csv(file_decode_csv, sep='\t')

   decoded_vectda = np.array(autoencoder.predict(x_test))
   decoded_vectda = pd.DataFrame(decoded_vectda)
   decoded_vectda.index = y_test
   file_decode_csv = "./output_output_layer_" + str(numlayer) + "_" + str(nummiddle) + "_test.txt"
   decoded_vectda.to_csv(file_decode_csv, sep='\t')

   file_model_encode = "./encode_" + str(numlayer) + "_" + str(nummiddle) + ".hdf5"
   file_model_decode = "./decode_" + str(numlayer) + "_" + str(nummiddle) + ".hdf5"
   encoder.save(file_model_encode)
   autoencoder.save(file_model_decode)

   original_hist = "./" + "history_folds_" + str(nfolds) + "_" + str(numlayer) + "_" + str(nummiddle) + "_valid.txt"
   copied_hist = dirda + "history_folds_" + str(nfolds) + "_valid.txt"
   shutil.copyfile( original_hist , copied_hist )
   original_middle = "./output_middle_layer_" + str(numlayer) + "_" + str(nummiddle) + "_valid.txt"
   copied_middle = dirda + "output_middle_layer_" + str(nfolds) + "_valid.txt"
   shutil.copyfile( original_middle , copied_middle )
   original_out = "./output_output_layer_" + str(numlayer) + "_" + str(nummiddle) + "_valid.txt"
   copied_out = dirda + "output_output_layer_" + str(nfolds) + "_valid.txt"
   shutil.copyfile( original_out , copied_out )
   original_out = "./output_output_layer_" + str(numlayer) + "_" + str(nummiddle) + "_test.txt"
   copied_out = dirda + "output_output_layer_" + str(nfolds) + "_test.txt"
   shutil.copyfile( original_out , copied_out )
   original_hdf1 = "./encode_" + str(numlayer) + "_" + str(nummiddle) + ".hdf5"
   copied_hdf1 = dirda + "encode_nfold_" + str(nfolds) + "_valid.hdf5"
   shutil.copyfile( original_hdf1 , copied_hdf1 )
   original_hdf2 = "./decode_" + str(numlayer) + "_" + str(nummiddle) + ".hdf5"
   copied_hdf2 = dirda + "decode_nfold_" + str(nfolds) + "_valid.hdf5"
   shutil.copyfile( original_hdf2 , copied_hdf2 )

   z_test = data_test
   z_index = data_test.index
   z_test = z_test.values
   z_re = z_test.reshape(-1,)
   decoded_test = np.array(autoencoder.predict(z_test))
   decoded_test_re = decoded_test.reshape(-1,)
   from sklearn.metrics import accuracy_score #***
   lossda = accuracy_score(z_re,np.rint(decoded_test_re).astype(np.float64)) #***
   print(lossda)

   if data_loss_value.empty == True:
       data_loss_value = pd.Series(lossda)
   else:
       data_loss_value = pd.concat([data_loss_value,pd.Series(lossda)])

   z_test = data_test
   z_index = data_test.index
   z_test = z_test.values
   z_re = z_test.reshape(-1,)
   decoded_test = np.array(autoencoder.predict(z_test))
   decoded_test_re = decoded_test.reshape(-1,)
   from sklearn.metrics import accuracy_score #***
   bce = tf.keras.losses.BinaryCrossentropy()
   lossda = bce(z_re,decoded_test_re.astype(np.float64))
   print(lossda)

   if data_loss_value_test.empty == True:
       data_loss_value_test = pd.Series(lossda.numpy())
   else:
       data_loss_value_test = pd.concat([data_loss_value_test,pd.Series(lossda.numpy())])

   for data in list_test:
       directoryda = "../../../../Sharing_SizeSame/{}/pdf.txt".format(data)
       cellcateg  = pd.read_csv("../../../../Sharing_SizeSame/{}/cell_categ_exc.txt".format(data),header = None,sep=",")
       layercateg = pd.read_csv("../../../../Sharing_SizeSame/{}/layer_categ.txt".format(data),header = None,sep=",")
       categ = (cellcateg)*8+(7-layercateg) # inh (deep-->surface)-->exc(deep-->surface)
       data1 = pd.read_csv(directoryda,header=None,sep=",")
       data1["CCat"] = categ[0].values.T
       data1 = data1.sort_values("CCat",ascending=False, kind="mergesort") #***
       data1 = data1.drop("CCat",axis=1)   # 1 --> 0
       data1 = data1.T
       data1["CCat"] = categ[0].values.T
       data1 = data1.sort_values("CCat",ascending=False)
       data1 = data1.drop("CCat",axis=1)   # 1 --> 0
       data1 = data1.iloc[0:100,0:100]
       data1 = data1.T 
       data1.index = range(100)
       data1.columns = range(100)
       encoded_vectda = np.array(encoder.predict(data1))
       encoded_vectda = pd.DataFrame(encoded_vectda)
       file_encode_csv = dirda + "./embedded_" + data + ".txt"
       encoded_vectda.to_csv(file_encode_csv, sep='\t')
       
   nfolds = nfolds + 1

data_loss_value = pd.DataFrame(data_loss_value)
data_loss_value.columns = ["accuracy"]
temp_file_name = dirda + "result_accuracy_five_fold.txt"
data_loss_value.to_csv(temp_file_name,sep="\t")

data_loss_value_test = pd.DataFrame(data_loss_value_test)
data_loss_value_test.columns = ["loss"]
temp_file_name = dirda + "result_loss_five_fold.txt"
data_loss_value_test.to_csv(temp_file_name,sep="\t")

datahistsum = pd.DataFrame()
datahistsum2 = pd.DataFrame()
for i in range(5): 
   itda = i + 1
   filehist = "./" + "history_folds_" + str(itda) + "_" + str(numlayer) + "_" + str(nummiddle) + "_valid.txt"
   datahist = pd.read_csv(filehist,index_col=0,sep=",")
   print(datahist["val_loss"])
   if i == 0:
      datahistsum = pd.Series(datahist["val_loss"].values)
      datahistsum2 = pd.Series(datahist["loss"].values)
   else:
      datahistsum = pd.concat([datahistsum,pd.Series(datahist["val_loss"].values)],axis=1)
      datahistsum2 = pd.concat([datahistsum2,pd.Series(datahist["loss"].values)],axis=1)
datahistsum = datahistsum.dropna(how="any")
datahistsum2 = datahistsum2.dropna(how="any")
datahistsum.columns = [1,2,3,4,5]
datahistsum2.columns = [1,2,3,4,5]
summda = datahistsum.T.describe()
summda  = summda.T +0.0000000000000000000000000000001
summda2 = datahistsum2.T.describe()
summda2 = summda2.T+0.0000000000000000000000000000001
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(summda2.index,np.log10( summda2["mean"]),color="b", ls="--", label="loss_training")
ax.plot(summda.index ,np.log10( summda["mean"]) ,color="r", ls="-", label="loss_valid")
ax.legend()
ax.set_title("loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
filenameda1 = dirda + "Loss_AE_SizeSame_valid.pdf"
filenameda2 = dirda + "Loss_AE_SizeSame_valid.eps"
filenameda3 = dirda + "Loss_AE_SizeSame_valid.jpeg"
plt.savefig(filenameda1)
plt.savefig(filenameda2)
plt.savefig(filenameda3)
plt.close()

datahistsum = pd.DataFrame()
datahistsum2 = pd.DataFrame()
for i in range(5): 
   itda = i + 1
   filehist = "./" + "history_folds_" + str(itda) + "_" + str(numlayer) + "_" + str(nummiddle) + "_valid.txt"
   datahist = pd.read_csv(filehist,index_col=0,sep=",")
   print(datahist["val_binary_accuracy"])
   if i == 0:
      datahistsum = pd.Series(datahist["val_binary_accuracy"].values)
      datahistsum2 = pd.Series(datahist["binary_accuracy"].values)
   else:
      datahistsum = pd.concat([datahistsum,pd.Series(datahist["val_binary_accuracy"].values)],axis=1)
      datahistsum2 = pd.concat([datahistsum2,pd.Series(datahist["binary_accuracy"].values)],axis=1)
datahistsum = datahistsum.dropna(how="any")
datahistsum2 = datahistsum2.dropna(how="any")
datahistsum.columns = [1,2,3,4,5]
datahistsum2.columns = [1,2,3,4,5]
summda = datahistsum.T.describe()
summda  = summda.T +0.0000000000000000000000000000001
summda2 = datahistsum2.T.describe()
summda2 = summda2.T+0.0000000000000000000000000000001
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(summda2.index,summda2["mean"],color="b", ls="--", label="accuracy_training")
ax.plot(summda.index ,summda["mean"] ,color="r", ls="-", label="accuracy_valid")
ax.legend()
ax.set_title("accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
filename1 = dirda + "Accuracy_AE_SizeSame_valid.pdf"
filename2 = dirda + "Accuracy_AE_SizeSame_valid.eps"
filename3 = dirda + "Accuracy_AE_SizeSame_valid.jpeg"
plt.savefig(filename1)
plt.savefig(filename2)
plt.savefig(filename3)
plt.close()
