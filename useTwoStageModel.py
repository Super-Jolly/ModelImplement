import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 比例
# or
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.layers import Lambda
from classEntity import Document
from Datasets import *
import keras.backend as K

from dataProcess import load_data, Initial
import numpy as np

def getPositionVec(dis):
    if dis <= -31:
        return 0
    elif -30 <= dis <= -21:
        return 1
    elif -20 <= dis <= -11:
        return 2
    elif -10 <= dis <= -6:
        return 3
    elif dis == -5:
        return 4
    elif dis == -4:
        return 5
    elif dis == -3:
        return 6
    elif dis == -2:
        return 7
    elif dis == -1:
        return 8
    elif dis == 0:
        return 9
    elif dis == 1:
        return 10
    elif dis == 2:
        return 11
    elif dis == 3:
        return 12
    elif dis == 4:
        return 13
    elif dis == 5:
        return 14
    elif 6 <= dis <= 10:
        return 15
    elif 11 <= dis <= 20:
        return 16
    elif 21 <= dis <= 30:
        return 17
    elif 31 <= dis:
        return 18


if __name__ == "__main__":
    # load data
    train_path = "data/train/"
    test_path = "data/test/"
    train_ = Datasets(filename=train_path)
    train_data = train_.features
    test_ = Datasets(filename=test_path)

    test_data = test_.features
    word_dict = initial.word_dict

    print("get the train data")
    train_array = list()
    train_label = list()
    train_postion1 = list()
    train_postion2 = list()
    train_entityNum = {}
    for feature in train_data:
        if feature['negative'] is False :
            if feature['label'] in [0, 1, 2, 3]:
                train_label.append(1)
            else:
                train_label.append(0)
            train_array.append(feature['all_sequence'])
            position_vec1 = []
            position_vec2 = []
            for i in range(len(feature['all_sequence'])):
                posEnt1 = feature['e1_pos']
                posEnt2 = feature['e2_pos']
                dis_1 = i - posEnt1
                dis_2 = i - posEnt2
                position_vec1.append(getPositionVec(dis_1))
                position_vec2.append(getPositionVec(dis_2))

            train_postion1.append(position_vec1)
            train_postion2.append(position_vec2)

    train_array = np.array(train_array)
    train_postion1 = np.array(train_postion1)
    train_postion2 = np.array(train_postion2)
    print(train_array.shape)

    print("get the test data")
    test_array = list()
    test_label = list()
    test_postion1 = list()
    test_postion2 = list()
    test_entityNum = {}
    test_act=list()

    for feature in test_data:
        if feature['negative'] is False :
            if feature['label'] in [0, 1, 2, 3]:
                test_label.append(1)
            else:
                test_label.append(0)
            test_act.append(feature['label'])
            test_array.append(feature['all_sequence'])

            position_vec1 = []
            position_vec2 = []
            for i in range(len(feature['all_sequence'])):
                posEnt1 = feature['e1_pos']
                posEnt2 = feature['e2_pos']
                dis_1 = i - posEnt1
                dis_2 = i - posEnt2
                position_vec1.append(getPositionVec(dis_1))
                position_vec2.append(getPositionVec(dis_2))
            test_postion1.append(position_vec1)
            test_postion2.append(position_vec2)

    test_array = np.array(test_array)
    test_postion1 = np.array(test_postion1)
    test_postion2 = np.array(test_postion2)

    print(test_array.shape)
    from keras.callbacks import ModelCheckpoint
    from keras.layers import Embedding
    import numpy as np
    from keras.layers import Dense, Input, Flatten, merge
    from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Bidirectional, Dropout,GRU,SimpleRNN
    from keras.models import Model
    import copy
    from keras.utils import np_utils
    from keras.layers.merge import concatenate
    from keras.layers.normalization import BatchNormalization
    from keras import regularizers
    from multi_head_attention import MultiHeadAttentionC
    from capsule_keras import *
    from keras.models import Model
    from keras.layers import *
    from keras import backend as K
    from MicroCalculate import *
    from sklearn.metrics import confusion_matrix

    word_embedding = Embedding(input_dim=word_dict.shape[0],
                               output_dim=200,
                               input_length=150,
                               weights=[word_dict],
                               trainable=False)
    position_embedding = Embedding(input_dim=19,
                                   output_dim=15,
                                   input_length=150,
                                   trainable=True)
    print('Training model.')
    input_word = Input(shape=(150,), dtype='int32', name='input_word')
    word_fea = word_embedding(input_word)

    input_pos1 = Input(shape=(150,), dtype='int32', name='input_pos1')
    pos_fea1 = position_embedding(input_pos1)

    input_pos2 = Input(shape=(150,), dtype='int32', name='input_pos2')
    pos_fea2 = position_embedding(input_pos2)

    Routings = 5
    dropout_p = 0.25
    n_head = 4
    x = concatenate([word_fea, pos_fea1, pos_fea2], axis=-1)
    x = GRU(128, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True)(x)
    x=MultiHeadAttentionC(head_num=n_head)([x,x,x])
    x = Conv1D(64, 3, activation='relu')(x)
    x = Conv1D(64, 5, activation='relu')(x)
    capsule = Capsule(num_capsule=2, dim_capsule=32, routings=Routings,share_weights=True)(x)
    preds = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)

    model = Model(inputs=[input_word,input_pos1, input_pos2], outputs=preds)
    model.summary()
    model.compile(loss=lambda y_true, y_pred: y_true * K.relu(0.9 - y_pred) ** 2 + 0.5 * (1 - y_true) * K.relu(y_pred - 0.1) ** 2,optimizer='adam',metrics=['accuracy'])

    train_label_cat = np_utils.to_categorical(train_label, 2) 
    test_backup = np.array(copy.deepcopy(test_label))
    test_label_cat = np_utils.to_categorical(test_label, 2)
    model.load_weights('model_saved/stage1ModelWeight.hdf5')

    predicted = model.predict([test_array, test_postion1, test_postion2], verbose=0)
    predicted = predicted.tolist()
    predict = []
    for i in range(len(predicted)):
        predict.append(predicted[i].index(max(predicted[i])))
    
    confusionMetrix = confusion_matrix(test_backup, predict, labels=[0, 1])
    print(confusionMetrix)


    finalTrue=[]
    finalPred=[]
    posIndex=[]
    for i in range(len(predict)):
        if predict[i]==0:
            finalTrue.append(test_act[i])
            finalPred.append(4)
        else:
            posIndex.append(i)

    test_arrayTwo = list()
    test_labelTwo = list()
    test_postion1Two = list()
    test_postion2Two = list()
    for i in posIndex:
        test_arrayTwo.append(test_array[i])
        test_labelTwo.append(test_act[i])
        test_postion1Two.append(test_postion1[i])
        test_postion2Two.append(test_postion2[i])

    test_arrayTwo  = np.array(test_arrayTwo)
    test_postion1Two  = np.array(test_postion1Two)
    test_postion2Two  = np.array(test_postion2Two)

    test_backupTwo = np.array(copy.deepcopy(test_labelTwo))
    test_label_catTwo = np_utils.to_categorical(test_labelTwo, 5)

    x = concatenate([word_fea, pos_fea1, pos_fea2], axis=-1)
    x = GRU(128, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True)(x)
    x=MultiHeadAttentionC(head_num=n_head)([x,x,x])
    x = Conv1D(64, 3, activation='relu')(x)
    x = Conv1D(64, 5, activation='relu')(x)
    capsule = Capsule(num_capsule=4, dim_capsule=32, routings=Routings,share_weights=True)(x)
    preds = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)

    model2 = Model(inputs=[input_word, input_pos1, input_pos2], outputs=preds)
    model2.summary()
    model2.compile(loss=lambda y_true, y_pred: y_true * K.relu(0.9 - y_pred) ** 2 + 0.5 * (1 - y_true) * K.relu(y_pred - 0.1) ** 2,optimizer='adam',metrics=['accuracy'])
    model2.load_weights('model_saved/stage2ModelWeight.hdf5')

    predictedTwo = model2.predict([test_arrayTwo,test_postion1Two, test_postion2Two], verbose=0)
    predictedTwo = predictedTwo.tolist()
    predict2 = []
    for i in range(len(predictedTwo)):
        predict2.append(predictedTwo[i].index(max(predictedTwo[i])))

    finalTrue = np.hstack((finalTrue, test_backupTwo))
    finalPred = np.hstack((finalPred, predict2))

    p, r, f = calculateMicroValue(y_pred=finalPred, y_true=finalTrue, labels=[0, 1, 2, 3, 4], filename='finalres.txt',filename2='finalprocess.txt')
    confusionMetrix2 = confusion_matrix(finalTrue, finalPred, labels=[0, 1, 2, 3, 4])
    print(confusionMetrix2)


