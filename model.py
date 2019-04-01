from keras.models import Model
from keras.layers import *
import preprocess as p
import os
from keras.callbacks import ModelCheckpoint

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# ~~~~~~~~~~~~~~~~~~~~~訓練資料前處理~~~~~~~~~~~~~~~~~~~~~~
# 讀檔
dataset = p.readFIle('data/2017to2018_power.csv')

# 將檔案切分成2017年 和2018年
Y2017_dataset = dataset.loc[dataset["日期"]<=20179999.0]
Y2018_dataset = dataset.loc[dataset["日期"]>=20179999.0]
Y2018_dataset.index = range(len(Y2018_dataset))

# 拿Mean std
dataset["日期"] = (dataset["日期"] % 10000.)
mean = dataset.mean(axis=0)
std = dataset.std(axis=0)
np.save("mean.npy",mean)
np.save("std.npy",std)

# 獲得訓練資料
trainData, trainTarget, trainIntput2 = p.getTraindata(Y2017_dataset, Y2018_dataset,mean,std)
size_in_1 = len(trainData[0])
size_in_2 = len(trainIntput2[0])
size_target = len(trainTarget[0])
print(np.size(trainIntput2))
# reshap
trainData = np.array(trainData).reshape(len(trainData),1,size_in_1)
trainTarget = np.array(trainTarget).reshape(len(trainTarget),1,size_target)
trainIntput2 = np.array(trainIntput2).reshape(len(trainIntput2),1,size_in_2)


# ~~~~~~~~~~~~~~~~~~~~~測試資料前處理~~~~~~~~~~~~~~~~~~~~~~
# 獲得測試資料
# 讀檔
test_dataset = p.readFIle('data/2018to2019_power.csv')

# 將檔案切分成2018年 和2019年
Y2018_dataset = test_dataset.loc[test_dataset["日期"] <= 20189999.0]
Y2019_dataset = test_dataset.loc[test_dataset["日期"] >= 20189999.0]
Y2019_dataset.index = range(len(Y2019_dataset))

# 獲得測試資料
testData, testTarget, testIntput2 = p.getTraindata(Y2018_dataset, Y2019_dataset,mean,std)

# reshap
testData = np.array(testData).reshape(len(testData),1,size_in_1)
testTarget = np.array(testTarget).reshape(len(testTarget),1,size_target)
testIntput2 = np.array(testIntput2).reshape(len(testIntput2),1,size_in_2)


input1 = Input(shape=(1,size_in_1))
input2 = Input(shape=(1,size_in_2))

x=Dense(64, activation='sigmoid')(input1)
x=Dense(16, activation='relu')(x)
model1=Dense(7, activation='relu')(x)

# model2 = input2
model2 = Dense(7, activation='relu')(input2)
model2 = Dense(7, activation='relu')(model2)

added = Add()([model1, model2])

model = Model(inputs=[input1, input2], outputs=added)
model.compile(optimizer='adam', loss=root_mean_squared_error)


filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]



model.fit([trainData,trainIntput2], trainTarget,
                epochs=1000,
                batch_size=20,
                shuffle=True,
                validation_data=([testData,testIntput2], testTarget),
                 callbacks=callbacks_list
                )


