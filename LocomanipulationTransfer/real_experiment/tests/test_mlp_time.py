import time
import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(48, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 12)
        )
    def forward(self, x):
        out = self.mlp(x)
        return out

model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

time_start = time.time()
x = np.random.ranf((1, 48))
x = torch.tensor(x, dtype=torch.float32)
result = model(x)
to_cpu = result.to('cpu')
print(to_cpu)
time_end = time.time()
print(time_end - time_start)