import os
import torch
import torchvision
from torchvision import transforms
import config
from models.unet import UNet
from loader import Loader

torch.manual_seed(2022)

#################
# Data loaders
#################
train_dataset = Loader(config.TRAIN_DIR, "train")
validation_dataset = Loader(config.VAL_DIR, "validation")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE)
valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.BATCH_SIZE)

device = torch.device(config.DEVICE)

#################
# Create Model
#################
model = UNet().to(device)
if config.PRETRAINED_WEIGHTS != None:
    model.load_state_dict(torch.load(config.PRETRAINED_WEIGHTS))
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
lossF = torch.nn.MSELoss()

#################
# Create output folders
#################
os.makedirs(config.TRAINING_OUTPUT_DIR, exist_ok=True)
visualization_dir = os.path.join(config.TRAINING_OUTPUT_DIR, "training_visualization")
os.makedirs(visualization_dir, exist_ok=True)

#################
# Training
#################
bestLoss = float("inf")
train_loss = 0
print("Start training")
for epoch in range(config.EPOCHS):
    print('============= EPOCH {}/{} ============='.format(epoch, config.EPOCHS))    
    model.train()
    elemCount = 0
    train_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = lossF(output, data)
        train_loss += loss.sum().item()
        elemCount += len(data)
        loss.backward()
        optimizer.step()
    train_loss /= elemCount

    model.eval()
    valid_loss = 0
    elemCount = 0
    with torch.no_grad():
        for data, _ in valid_loader:
            data = data.to(device)
            output = model(data)

            # Regularly show reconstruction on first image of the validation set to check progress
            if epoch % config.VAL_VISUALIZATION == 0 : 
                ref = data[0].detach().cpu().squeeze_(0)
                img = output[0].detach().cpu().squeeze_(0)
                torchvision.utils.save_image(ref, os.path.join(visualization_dir, str(epoch) + "_ref.png"))
                torchvision.utils.save_image(img, os.path.join(visualization_dir, str(epoch) + "_pred.png"))

            valid_loss += lossF(output, data).sum().item()
            elemCount += len(data)

    valid_loss /= elemCount

    print("Epoch " + str(epoch) + "  :  train_Loss=" + str(train_loss) + "    val_Loss=" + str(valid_loss))
    if valid_loss < bestLoss :
        bestLoss = valid_loss
        torch.save(model.state_dict(), os.path.join(config.TRAINING_OUTPUT_DIR,"best.pt"))
        print("Model saved")