# Comparison for training CNN models using Tensor Cores with PyTorch

This is a simple comparison for training CNN models using Tensor Cores with PyTorch framework. Although there are sample scripts on the official site, there should be a lot of differences in real world applications.

## Hardware and Software

GPU: Tesla T4

Framework: PyTorch 1.11

nvidia Driver: 470.57.02

cuda: 11.4

## Performance

### Sample Code without using Tensor Cores

```
for epoch in range(epochs):
    for input, target in zip(data, targets):
        output = net(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
```

```
Default precision:
Total execution time = 7.020 sec
Max memory used by tensors = 1350681600 bytes
```

### Sample Code using Tensor Cores

```
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
```

```
Mixed precision:
Total execution time = 2.524 sec
Max memory used by tensors = 1577232384 bytes
```

The one using Tensor Cores used even more memory, but the execution time is half of that not using Tensor Cores.

### MNIST with CNN without using Tensor Cores (Batch size 512)

```
trainloader = torch.utils.data.DataLoader(trainset,batch_size=512,shuffle=True,num_workers=2)

for epoch in range(20):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()

        outputs = net(inputs.to(device))
        loss = loss_fn(outputs, labels.to(device))
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    scheduler.step()
```

```
Total execution time = 107.228 sec
Max memory used by tensors = 417359872 bytes
```

### MNIST with CNN using Tensor Cores (Batch size 512)

```
trainloader = torch.utils.data.DataLoader(trainset,batch_size=512,shuffle=True,num_workers=2)

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for epoch in range(20):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if i % 500 == 499:
            print('[{:d}, {:5d}] loss: {:.3f}'
                    .format(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    
    scheduler.step()
```

```
Total execution time = 104.948 sec
Max memory used by tensors = 175422976 bytes
```

### MNIST with CNN using Tensor Cores (Batch size 1024)

```
trainloader = torch.utils.data.DataLoader(trainset,batch_size=1024,shuffle=True,num_workers=2)

use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for epoch in range(20):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if i % 500 == 499:
            print('[{:d}, {:5d}] loss: {:.3f}'
                    .format(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    
    scheduler.step()
```

```
Total execution time = 100.529 sec
Max memory used by tensors = 320654848 bytes
```

Using Tensor Cores did not significantly decrease the execution time, conversely the memory decreased over by half, even if after doubling the batch size, the maximum memory used is also less than the original one, with further a little bit decrease in execution time.

For some reasons, maybe other bottlenecks in the model, using Tensor Cores did not improve the training speed, but less memory means a larger batch size can be used for training, this is good because usually training a real world large model requires a lot of GPU ram, it would be much easier to deploy those models in a small VM with not much GPU ram.
