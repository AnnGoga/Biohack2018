##from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Activation, Dense, Dropout, BatchNormalization
from keras.layers import LeakyReLU, ThresholdedReLU, PReLU, ELU
from keras.models import Sequential
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

col = len([c for c in train_df.columns if c != 'stain'])


def nn(x_train, y_train, x_test, y_test, actinput, op, los, k1, k2):
    model = Sequential()
    model.add(Dense(k1, input_dim=col))
    model.add(BatchNormalization())
    # try activation function
    try:
        model.add(Activation(actinput))
    except:
        model.add(actinput())

    model.add(Dropout(0.5))

    model.add(Dense(k2,
                    # activity_regularizer=l2(0.01)
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
                        epochs=1,
                        # batch_size=int(col / 10),
                        verbose=1,
                        validation_data=(x_test, y_test),
                        shuffle=True,
                        # callbacks=[reduce_lr]
                        )

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
    plt.show()


# Available activation functions
activate_functions = [
    # 'elu',
    # 'selu',
    # 'softplus',
    # 'softsign',
    # 'softmax',
    # 'relu',
    # 'tanh',
    # 'sigmoid',
    # 'hard_sigmoid',
    LeakyReLU,
    ThresholdedReLU,
    PReLU,
    ELU,
]

# lost function to go with
loss_functions = [
    'binary_crossentropy',
    # 'categorical_hinge',
    'categorical_crossentropy',
    # 'sparse_categorical_crossentropy',
]

# Optimizers
optimizers = [
    # 'sgd',
    # 'adam',
    'rmsprop',
    # 'adagrad',
    # 'adadelta',
    # 'adamax',
    # 'nadam',
]

# def create_smaller():
#     model = Sequential()
#     model.add(Dense(30, input_dim=60, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
#
# x = pd.concat([train_df, test_df]).as_matrix([c for c in train_df.columns if c != 'stain'])
# y = pd.concat([train_df, test_df]).as_matrix(['stain'])
#
# estimators = [
#     ('standardize', StandardScaler()),
#     ('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0))
# ]
# pipeline = Pipeline(estimators)
# # fit = pipeline.fit(x_train, y_train)
# # score = pipeline.score(x_test, y_test)
# # print(score)
# # exit(0)
#
# # kfold = StratifiedKFold(n_folds=10, shuffle=True, random_state=seed)
# results = cross_val_score(pipeline, x, y)
# print("Smaller: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
#
# exit(0)

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

t = 0
print("Start NN")
attempt = 0
min_acc = 100
min_model = None
min_history = None
while attempt < 1000:
    for op in optimizers:
        for hl in [8, 64]:
            for af in activate_functions:
                for lf in loss_functions:
                    print('OP', op, "AF", af, 'LF', lf)
                    score, model, history = nn(x_train, y_train, x_test, y_test, af, op, lf, hl, hl)
                    print("Attempt", attempt, 'Score', score)
                    attempt += 1
                    if np.isfinite(score[0]):
                        min_acc = score[1]
                        min_model = model
                        min_history = history
                        print("NN, attempts", attempt, "loss", score[0], 'acc', score[1])
    break

if min_model is not None:
    name = 'best_model'
    # min_model.save('result/' + name + '.h5')
    vis(min_history, 'result/' + name, name + '_' + str(min_acc))

print('FAILED to optimize NN, attempts', attempt)
