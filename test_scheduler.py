import torch
from torch import nn

lr = 0.1

model = nn.Linear(10, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, verbose=True)

for epoch in range(10):
    optimizer.step()
    scheduler.step()
    # print(f"LR: {optimizer.state_dict()['param_groups'][0]['lr']}")
    print(f"LR: {scheduler.get_last_lr()[0]}")