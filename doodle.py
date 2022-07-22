import torch



a = torch.LongTensor([[0, 1, 2], [1, 2, 3]])
b = a.unsqueeze(1).repeat(1, 3, 1)

print(b)