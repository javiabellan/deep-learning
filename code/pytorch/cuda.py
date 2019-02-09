import torch

x = torch.rand(5, 3)
y = torch.rand(5, 3)

print(x)
print(y)

if torch.cuda.is_available():
    print("Sumando tensores en GPU...")

    x = x.cuda()
    y = y.cuda()
    sum = x + y

    print(sum)