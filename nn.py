import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from preprocessing import *


# label
Y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 1])
# target = [0, 1, 2, 4, 8, 9]  # 1-9-10, 2-3-5
target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

X = list()

for prefix, data in get_data(packages["RCTD"]):
    X.append(data)
    # if prefix == "P4_pre" or  prefix == "P7_pre" or prefix == "P8_pre":
    # if prefix != "P1_pre" and prefix != "P9_pre" and prefix != "P10_pre":
    print(prefix, data)


X = np.array(X)


good = [x for i, x in enumerate(X) if i in [0, 8, 9]]
mid = [x for i, x in enumerate(X) if i in [3, 6, 7]]
bad = [x for i, x in enumerate(X) if i in [1, 2, 4, 5]]


def pop(x):
    x.pop(0)
    x.pop(1)


print("good:")
print(np.array(good))
print("mid:")
print(np.array(mid))
print("bad:")
print(np.array(bad))

temp = []
temp.extend(good)
temp.extend(mid)
temp.extend(bad)


df = pd.DataFrame(np.array(temp))
df.insert(0, 'id', [1,9,10,4,7,8,2,3,5,6])

df.to_csv("result.csv", index=False)


quit()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))

        return x


# model, loss func, optimzer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    loss.backward()
    optimizer.step()


# plot!
a = 2
xx, yy = np.meshgrid(np.linspace(-a, a, 500), np.linspace(-a, a, 500))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
Z = model(grid).detach().numpy().reshape(xx.shape)

# 绘制点和决策边界
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=30)
plt.contour(xx, yy, Z, levels=[0.5], linewidths=1, colors="black")
plt.xlabel("immuneEPI+Differential")
plt.ylabel("CYCLING+STRESS+Intermedian+EMT")
plt.title("nn")
plt.axis("equal")  # 确保轴的比例相等

for i, (x, y) in enumerate(X):
    plt.text(x, y, str(i + 1), fontsize=9, ha="left")

plt.savefig("result.png", dpi=300)
plt.close()
