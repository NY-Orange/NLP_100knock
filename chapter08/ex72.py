import torch
import torch.nn as nn
from ex71 import SLPNet

def main():
    word_vec_dim = 300
    label_num = 4

    X_train = torch.load("./data/X_train.pt")
    y_train = torch.load("./data/y_train.pt")
    model = SLPNet(word_vec_dim, label_num)

    criterion = nn.CrossEntropyLoss()

    l_1 = criterion(model(X_train[:1]), y_train[:1])
    model.zero_grad()
    l_1.backward()

    print("損失 : {:.3f}".format(l_1))
    print("勾配 :\n{}".format(model.fc.weight.grad))

    l = criterion(model(X_train[:4]), y_train[:4])
    model.zero_grad()
    l.backward()

    print("損失 : {:.3f}".format(l))
    print("勾配 :\n{}".format(model.fc.weight.grad))

if __name__ == "__main__":
    main()