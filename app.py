from keras.models import load_model
from keras.models import Model
from keras.layers import *
import preprocess as p

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
# 讀檔
dataset = p.readFIle('data/2017to2018_power.csv')
mean = np.load('mean.npy')
std = np.load('std.npy')
# dataset = (dataset - mean) / std


# 將檔案切出2018年
Y2018_dataset = dataset.loc[dataset["日期"]>=20179999.0]
Y2018_dataset.index = range(len(Y2018_dataset))

# 找出去年的3個禮拜 normalize
mask1 = Y2018_dataset["日期"]>=20180326.0
mask2 = Y2018_dataset["日期"]<=20180415.0
testData = Y2018_dataset.loc[mask1&mask2]
testData = p.normalizeTest(testData,mean,std)

td = []  # 要input的input1
td.extend(testData[:,0:5].flatten())
# td.extend(testData[:,-1])

# # 前一個禮拜的尖峰電量
# preWeek_e = [28756,29140,30093,29673,25810,24466,28535]
# mean_e = mean[2]
# std_e = std[2]
# preWeek_e_n = (preWeek_e - mean_e) / std_e  # 正規劃
#
# td.extend(preWeek_e_n)
#
# 前一個禮拜的日期
preWeek_d = [326,327,328,329,330,331,401]
mean_d = mean[0]
std_d = std[0]
preWeek_d_n = (preWeek_d - mean_d) / std_d  # 正規劃

td.extend(preWeek_d_n)

# input2 => 上個禮拜的尖峰電量
testIntput2  = [28756,29140,30093,29673,25810,24466,28535]

# 讀訓練好的model
model = load_model(r'model/weights.best.hdf5', custom_objects={'root_mean_squared_error': root_mean_squared_error})
# model = load_model(r'model/weights.best.hdf5')

predic = model.predict([[[td]],[[testIntput2]]])

with open("submission.csv",'w') as f:
    f.write("date,peak_load(MW)"+"\n")
    f.write("20190402, "+str(predic[0][0][0])+"\n")
    f.write("20190403, "+str(predic[0][0][1])+"\n")
    f.write("20190404, "+str(predic[0][0][2])+"\n")
    f.write("20190405, "+str(predic[0][0][3])+"\n")
    f.write("20190406, "+str(predic[0][0][4])+"\n")
    f.write("20190407, "+str(predic[0][0][5])+"\n")
    f.write("20190408, "+str(predic[0][0][6])+"\n")

