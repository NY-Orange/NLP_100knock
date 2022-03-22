import torch
import torch.nn as nn

class SLPNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        # 正規乱数で重みを初期化
        nn.init.normal_(self.fc.weight, 0.0, 1.0)
    
    def forward(self, x):
        x = self.fc(x)
        return x



def main():
    word_vec_dim = 300
    label_num = 4

    X_train = torch.load("./data/X_train.pt")
    model = SLPNet(word_vec_dim, label_num)
    
    y_hat_1 = torch.softmax(model(X_train[:1]), dim=-1)
    print("y_hat_1 :\n", y_hat_1)

    y_hat = torch.softmax(model.forward(X_train[:4]), dim=-1)
    print("y_hat :\n", y_hat)



if __name__ == "__main__":
    main()