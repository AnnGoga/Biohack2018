##from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Activation, Dense, Dropout, BatchNormalization
from keras.layers import LeakyReLU, ThresholdedReLU, PReLU, ELU
from keras.models import Sequential
from keras.utils import to_categorical

##
##
##import numpy
##import pandas
##from keras.models import Sequential
##from keras.layers import Dense
##from keras.wrappers.scikit_learn import KerasClassifier
##from keras.utils import np_utils


train_df = pd.read_csv('result/train.csv', sep=',')
print(train_df.head(1))
print('TEST stains', set(train_df['stain']))
test_df = pd.read_csv('result/test.csv', sep=',')
print(test_df.head(1))
print('TRAIN stains', set(test_df['stain']))
all_categories = set(list(train_df['stain']) + list(test_df['stain']))
print('ALL categories', all_categories)

x_train = train_df.as_matrix([c for c in train_df.columns if c != 'stain'])
y_train = train_df.as_matrix(['stain'])

x_test = test_df.as_matrix([c for c in test_df.columns if c != 'stain'])
y_test = test_df.as_matrix(['stain'])

seed = 7
np.random.seed(seed)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def nw(x_train, y_train, x_test, y_test, actinput, op, los, k1, k2):
    model = Sequential()
    model.add(Dense(k1, input_dim=len([c for c in train_df.columns if c != 'stain'])))
    model.add(BatchNormalization())
    # try activation function
    try:
        model.add(Activation(actinput))
    except:
        model.add(actinput())

    model.add(Dropout(0.45))

    model.add(Dense(k2,
                    ##              activity_regularizer=l2(0.01)
                    ))
    model.add(BatchNormalization())
    try:
        model.add(Activation(actinput))
    except:
        model.add(actinput())
    model.add(Dense(len(y_train[0])))
    try:
        model.add(Activation(actinput))
    except:
        model.add(actinput())
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, verbose=1)
    ##    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    ##    loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(optimizer=op,
                  loss=los,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        # epochs=50,
                        ##              batch_size = int(len(col)/10),
                        verbose=1,
                        validation_data=(x_test, y_test),
                        shuffle=True,
                        callbacks=[reduce_lr])

    score = model.evaluate(x_test, y_test)
    return score, model, history


def vis(history, file, name):
    ##    plt.figure()
    ##    plt.plot(history.history['loss'])
    ##    plt.plot(history.history['val_loss'])
    ##    plt.title('model loss')
    ##    plt.ylabel('loss')
    ##    plt.xlabel('epoch')
    ##    plt.legend(['train', 'test'], loc='best')
    ##    plt.show()

    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(name)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    fig.savefig(file + '.png')


# Available activation functions
activate = ['elu', 'selu', 'softplus', 'sofsign', 'relu',
            'tanh', 'sigmoid', 'hard_sigmoid', 'softmax', LeakyReLU, ThresholdedReLU, PReLU, ELU]

# lost function to go with
los = ['binary_crossentropy']

# Optimizers
opt = ['rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam', 'sgd']

# save each best model for setting = n
score = [0, -1]
# max hidden units
m = 15
# max count of repeat for each setting
rep = 3
d = 0
# min accur and safe of repeat cycle
accur = 0.6
cycle = 15
h = 0
# epoch
ep = 100

# for l in los:
#     train_score, score, model, history = nw(x_train, y_train, x_test, y_test, LeakyReLU, LeakyReLU, opt[0], l, 8, 8)
#     name = 'MLP-' + str(8) + '-' + str(8) + '-' + str('LeakyReLU') + '-' + str('LeakyReLU') + '-' + str(
#         opt[0]) + '-' + l
#     ##    model.save('NW/result/'+name+'.h5')
#     vis(history, 'NW/result/' + name, name + '_' + str(score[1]) + '_' + str(train_score))
#     print(score[1])
#
#     ##    d=model.predict_classes(x_exp)
#     print(d)

# opt = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]
t = 0
print("Start NN")
attempt = 0
while attempt < 100:
    score, model, history = nw(x_train, y_train, x_test, y_test, LeakyReLU, opt[0], los[0], 8, 8)
    print("Attempt", attempt, 'Score', score)
    attempt += 1
    if not np.isfinite(score[0]):
        continue

    print("DONE NN, attempts", attempt, "score", score)
    name = 'MLP-' + str(8) + '-' + str(8) + '-' + str('LeakyReLU') + \
           '-' + str('LeakyReLU') + '-' + str(opt[0]) + '-' + los[0]
    model.save('result/' + name + '.h5')
    vis(history, 'result/' + name, name + '_' + str(score[1]))
    exit(0)

print('FAILED to optimize NN, attempts', attempt)
##with open ('hh.csv',tag) as data,open('hh_best.csv','a') as dat:
##    print('start')
##    data.write(';'.join(['MLP','IN_act','OUT_act','OPTIM','Loss','Hidden_1','Hidden_2','Accuracy_Train','Accuracy_Test','MLP_N'])+'\n')
##    dat.write(';'.join(['MLP','IN_act','OUT_act','OPTIM','Loss','Hidden_1','Hidden_2','Accuracy_Train','Accuracy_Test','MLP_N'])+'\n')

##for i in [activat[t]]:
##    for j in activat:
##        n=0
##        for op in opt:
##            for l in los:
##                for k2 in range(2,m):
##                    
##                    k1=25
##                    
##                    d,h=0,0
##                    while d<rep:
##                        train_score,score, model,history=nw(x_train,y_train,x_test,y_test,i,j,op,l,k1,k2,g)
##                        f1,f2=i,j
##                        if not isinstance(i,str):
##                            f1=i.__name__
##                        if not isinstance(j,str):
##                            f2=j.__name__
##                        name='MLP-'+str(k1)+'-'+str(k2)+'-'+str(f1)+'-'+str(f2)+'-'+str(op)+'-'+str(d)
##                        with open ('hh.csv','a') as data:
##                            data.write(';'.join([name,str(f1),str(f2),str(op),str(l),str(k1),str(k2),str(train_score),str(score[1]),str(d)])+'\n')
##                        
##                        print('************************'+str(score[1])+'/'+str(train_score)+'_In:'+str(f1)+'_Out:'+str(f2)+'_'+str(k1)+'-'+str(k2)+str(op)+'*******************')
##                        
##                        if score[1]>accur and train_score>accur:
##                        
##                            with open ('hh_best.csv','a') as data:
##                                data.write(';'.join([name,str(f1),str(f2),str(op),str(l),str(k1),str(k2),str(train_score),str(score[1]),str(d)])+'\n')
##                            d+=1
##                            h=0
####                                if score[1]>n:
####                                n=score[1]
##                            print('saving')
##                            print(score[1])
##                            model.save('NW/result/'+name+'.h5')
##                            vis(history,'NW/result/'+name,name+'_'+str(score[1])+'_'+str(train_score))
##                        else:
##                            h+=1
####                            
##                            
##                        if h>cycle:

##                                    log.write(';'.join([name,str(d)])+'\n')
##                            break



##                                print('__*****____',score, i,j,op,l,k1,k2)


##           



##            
##model.save('NW/my_model.h5')
