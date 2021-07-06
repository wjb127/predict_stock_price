import torch
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style

import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 랜덤 시드값
torch.manual_seed(77)

# 활성화 함수
def sigmoid(x):
    return torch.div(torch.tensor(1.0), torch.add(torch.tensor(1.0), torch.exp(torch.negative(x))))

def sigmoid_prime(x):
    return torch.mul(sigmoid(x), torch.subtract(torch.tensor(1.0), sigmoid(x)))

def sigmoid_inverse(x):
    return torch.negative(torch.log(torch.sub(torch.div(torch.tensor(1.0),x),torch.tensor(1.0))))

# 기울기 저장하는 파일
#FILE_NAME = 'gradient.txt'
#f = open(FILE_NAME, 'w')

# 트레이닝 데이터
df = pd.read_excel('train/train_naver.xlsx')
data2 = np.array(df)

#print(len(data2[0]))

# 인공신경망의 인풋 아웃풋
# 길이 그대로 가져오기
inputlen = 9
datalen = len(data2[0])
outputlen = 351

# 인풋과 레이블 설정
X = []
Y = []


for i in range(0,datalen - inputlen):


    X.append(data2[0][i:i+inputlen])
    Y.append(data2[0][i+inputlen])

X = [np.array(X)]





dtype = torch.float32
D_in, H, D_out = inputlen, 500, outputlen


try:
    w1re = pd.read_excel('wbdata/w1_naver10.xlsx', index_col=None)
    w1np = np.array(w1re)
    w1ts = torch.Tensor(w1np)

    w2re = pd.read_excel('wbdata/w2_naver10.xlsx', index_col=None)
    w2np = np.array(w2re)
    w2ts = torch.Tensor(w2np)

    b1re = pd.read_excel('wbdata/b1_naver10.xlsx', index_col=None)
    b1np = np.array(b1re)
    b1ts = torch.Tensor(b1np)

    b2re = pd.read_excel('wbdata/b2_naver10.xlsx', index_col=None)
    b2np = np.array(b2re)
    b2ts = torch.Tensor(b2np)

    # A weight and a bias for input nodes
    w1 = w1ts
    b1 = b1ts

    # A weight and a bias for hidden nodes
    w2 = w2ts
    b2 = b2ts

    if(w1.shape != torch.randn(D_in, H, dtype=dtype, requires_grad=True).shape or b1.shape != torch.randn(1, H, dtype=dtype, requires_grad=True).shape
            or w2.shape != torch.randn(H, D_out, dtype=dtype, requires_grad=True).shape or b2.shape != torch.randn(1, D_out, dtype=dtype, requires_grad=True).shape):

        raise Exception

    # print(w1.shape == torch.randn(D_in, H, dtype=dtype, requires_grad=True).shape)
    # print(b1.shape)
    # print(w2.shape)
    # print(b2.shape)
    # print(torch.randn(D_in, H, dtype=dtype, requires_grad=True).shape)



except:
    # A weight and a bias for input nodes
    w1 = Variable(torch.randn(D_in, H, dtype=dtype, requires_grad=True)) * np.sqrt(1. / D_in)
    b1 = Variable(torch.randn(1, H, dtype=dtype, requires_grad=True)) * np.sqrt(1. / D_in)

    # A weight and a bias for hidden nodes
    w2 = Variable(torch.randn(H, D_out, dtype=dtype, requires_grad=True)) * np.sqrt(1. / H)
    b2 = Variable(torch.randn(1, D_out, dtype=dtype, requires_grad=True)) * np.sqrt(1. / H)

full_epoch = 100

learning_rate = 0.1
for epoch in range(full_epoch):
    corrects = 0
    for i in range(0,datalen - inputlen):


        x = torch.Tensor([X[0][i]])
        y = np.array([Y[i]])

        new_y = y + (outputlen-1)//2
        y_onehot = torch.zeros((1, outputlen))

        # new_y 영역이 벗어나는 경우 처리 필요

        y_onehot[0, new_y] += 1


        z1 = torch.add(torch.mm(x, w1), b1)
        a1 = sigmoid(z1)
        z2 = torch.add(torch.mm(a1, w2), b2)
        a2 = sigmoid(z2)

        diff = a2 - y_onehot


        # backward pass
        d_z2 = torch.mul((a2 - y_onehot), sigmoid_prime(z2))
        d_b2 = torch.mul((a2 - y_onehot), sigmoid_prime(z2))

        #print(diff.shape,sigmoid_prime(z2).shape)

        d_w2 = torch.mm(torch.transpose(a1, 0, 1), torch.mul((a2 - y_onehot), sigmoid_prime(z2)))

        d_a1 = torch.mm(torch.mul((a2 - y_onehot), sigmoid_prime(z2)), torch.transpose(w2, 0, 1))
        d_z1 = torch.mul(torch.mm(torch.mul((a2 - y_onehot), sigmoid_prime(z2)), torch.transpose(w2, 0, 1)),
                         sigmoid_prime(z1))
        d_b1 = torch.mul(torch.mm(torch.mul((a2 - y_onehot), sigmoid_prime(z2)), torch.transpose(w2, 0, 1)),
                         sigmoid_prime(z1))
        d_w1 = torch.mm(torch.transpose(x, 0, 1),
                        torch.mul(torch.mm(torch.mul((a2 - y_onehot), sigmoid_prime(z2)), torch.transpose(w2, 0, 1)),
                                  sigmoid_prime(z1)))


        # weight update
        w1 -= learning_rate * d_w1
        w2 -= learning_rate * d_w2
        b1 -= learning_rate * d_b1
        b2 -= learning_rate * d_b2



        amaxa2 = torch.argmax(a2).numpy()
        yt = torch.Tensor(new_y)

        ytt = int(yt.numpy()[0])

        if amaxa2 == ytt:

            corrects += 1





        #if i % 10000 == 0:
        print("Epoch {}: {}/{}".format(epoch + 1, i+1, datalen - 9),end="")
        if amaxa2 != ytt:
            print(" Error!")
        else:
            print()

    print("Epoch {}, Accuracy: {:.3f}".format(epoch + 1, corrects / (datalen - 9) ))
    #if(corrects / len(data2) == 1):

        #print(w1,w2,b1,b2)

raw = pd.DataFrame(np.array(w1))
raw.to_excel(excel_writer='wbdata/w1_naver10.xlsx',index=None)
raw = pd.DataFrame(np.array(w2))
raw.to_excel(excel_writer='wbdata/w2_naver10.xlsx',index=None)
raw = pd.DataFrame(np.array(b1))
raw.to_excel(excel_writer='wbdata/b1_naver10.xlsx',index=None)
raw = pd.DataFrame(np.array(b2))
raw.to_excel(excel_writer='wbdata/b2_naver10.xlsx',index=None)


df2 = pd.read_excel('test/test_naver.xlsx')
data3 = np.array(df2)

X2 = []

X2.append(data3[0][0:datalen-1])
X2 = [np.array(X2)]
x2 = torch.Tensor([X2[0][0]])

for i in range(0,5):



    # 다음을 예측해야할 9일간의 주가 데이터
    print(np.array(x2[0]))
    # 신경망 구성
    z1 = torch.add(torch.mm(x2, w1), b1)
    a1 = sigmoid(z1)
    z2 = torch.add(torch.mm(a1, w2), b2)
    a2 = sigmoid(z2)

    # 결과 레이블
    if(torch.sum(a2[0][0:(outputlen)//2 - 1]) < torch.sum(a2[0][(outputlen)//2 + 1:outputlen-1])):
        print("상승")
    elif torch.sum(a2[0][0:(outputlen)//2 - 1]) > torch.sum(a2[0][(outputlen)//2 + 1:outputlen-1]):
        print("하락")
    else:
        print("횡보")

    print("상승 확률 : ", round(float(100 * torch.sum(a2[0][(outputlen) // 2 + 1:outputlen]) / torch.sum(a2)), 2))
    print("하락 확률 : ", round(float(100 * torch.sum(a2[0][0:(outputlen) // 2]) / torch.sum(a2)), 2))
    print("횡보 확률 : ", round(float(100 * torch.sum(a2[0][(outputlen) // 2 ]) / torch.sum(a2)), 2))
    # 결과물의 최대값 보기 = 의미하는 값
    amaxa2 = torch.argmax(a2).numpy()

    # 계산 결과물 : print(a2)
    # 예측 결과
    print("주가 변화 : ",amaxa2 - outputlen//2)
    print()

    x2 = list(x2[0])
    del x2[0]
    x2.append(torch.Tensor([amaxa2 - (outputlen)//2]))
    x2 = [x2]
    x2 = torch.Tensor(x2)
    # x2 = [x2]
    # x2 = torch.Tensor(x2)
    # #print(x2)
    # # 신경망 구성
    # z1 = torch.add(torch.mm(x2, w1), b1)
    # a1 = sigmoid(z1)
    # z2 = torch.add(torch.mm(a1, w2), b2)
    # a2 = sigmoid(z2)

    # 결과 레이블
    # if (torch.sum(a2[0][0:(outputlen) // 2 - 1]) < torch.sum(a2[0][(outputlen) // 2 + 1:outputlen - 1])):
    #     print("2차 상승 확률", round(float(100 * torch.sum(a2[0][(outputlen) // 2 + 1:outputlen - 1]) / torch.sum(a2)), 2))
    # elif torch.sum(a2[0][0:(outputlen) // 2 - 1]) > torch.sum(a2[0][(outputlen) // 2 + 1:outputlen - 1]):
    #     print("2차 하락 확률", round(float(100 * torch.sum(a2[0][0:(outputlen) // 2 - 1]) / torch.sum(a2)), 2))
    # else:
    #     print("2차 횡보")
    #
    # # 결과물의 최대값 보기 = 의미하는 값
    # amaxa2 = torch.argmax(a2).numpy()
    #
    # # 계산 결과물 : print(a2)
    # # 예측 결과
    # print("주가 변화 : ",amaxa2 - (outputlen) // 2)





# 학습지식 정리
# 이상적인 0부터 60까지 레이블에 역함수 적용
# 시그모이드 역함수 : z1 = sigmoid_inverse(x)
# b2 빼고 w2의 역행렬 곱하기
# 시그모이드 역함수
# b1 빼고 w1 역행렬 곱하기


###############################################################################

# U2, s2, V2 = np.linalg.svd(w2, full_matrices = True)
# S2 = np.zeros(w2.shape)
# for i in range(len(s2)):
#     S2[i][i] = s2[i]
# w2in = torch.Tensor(np.dot(U2, np.dot(S2, V2)))* np.sqrt(1. / U2.shape[0])
#
# U1, s1, V1= np.linalg.svd(w1, full_matrices = True)
# S1 = np.zeros(w1.shape)
# for i in range(len(s1)):
#     S1[i][i] = s1[i]
# w1in = torch.Tensor(np.dot(U1, np.dot(S1, V1)))* np.sqrt(1. / U1.shape[0])
#
#
# x = torch.zeros((1,outputlen))
#
# x += 0.01
# x[0,outputlen-1] += 0.98
# #print(a2)
#
# x = a2
# #print(x)
# z1 = sigmoid_inverse(x)
# #print(z1)
# a1 = torch.mm(torch.sub(z1,b2),torch.transpose(w2in,0,1))
# #print(a1)
# z2 = sigmoid_inverse(a1)
# #print(z2)
# a2 = torch.mm(torch.sub(z2,b1),torch.transpose(w1in,0,1))


#amaxa2 = torch.argmax(a2).numpy()
#print(x)
#print(a2)
#print(amaxa2 - outputlen // 2)

###############################################################################

bq = []
for k in range(0,outputlen):
    x = torch.zeros((1,outputlen))

    x += 0.01
    x[0,k] += 0.98
    #print(a2)

    #x = a2
    #print(k,end=" : ")

    z1 = sigmoid_inverse(x)
    a1 = torch.mm(z1, torch.transpose(w2,0,1))
    a1 -= np.min(np.array(a1))
    a1 /= np.max(np.array(a1))
    a1 *= 0.98
    a1 += 0.01

    z2 = sigmoid_inverse(a1)
    a2 = torch.mm(z2, torch.transpose(w1,0,1))
    a2 -= np.min(np.array(a2))
    a2 /= np.max(np.array(a2))
    a2 *= 0.98
    a2 += 0.01


    #amaxa2 = torch.argmax(a2).numpy()
    #print(x)
    bq.append(np.array(a2[0]))
    #print(np.array(a2[0]))

raw = pd.DataFrame(np.array(bq))
raw.to_excel(excel_writer='bqdata/backquery_naver10.xlsx', index=None)
    # for j in range(0,len(a2[0])):
    #
    #     print('%6s' %format(np.array(a2[0][j]),'.3'),end=" ")
    # print()
    #print(amaxa2 - outputlen // 2)

###############################################################################

# x2 = torch.Tensor([X2[0][0]])
# z1 = torch.add(torch.mm(x2, w1ts), b1ts)
# a1 = sigmoid(z1)
# z2 = torch.add(torch.mm(a1, w2ts), b2ts)
# a2 = sigmoid(z2)
#
# amaxa2 = torch.argmax(a2).numpy()
# print(x2)
# print(amaxa2 - (outputlen - 1) / 2)


# 아웃풋 파일에 쓰기
# f.write(str(w1))
# f.write("\n")
# f.write(str(b1))
# f.write("\n")
# f.write(str(w2))
# f.write("\n")
# f.write(str(b2))
# f.write("\n")
# f.close()