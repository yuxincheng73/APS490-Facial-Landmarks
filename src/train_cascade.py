import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import config
import utils
import numpy as np

from model import FaceKeypointModel, FaceKeypointModelStage2
from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm

matplotlib.style.use('ggplot')

# model 
stage1_model = FaceKeypointModel().to(config.DEVICE)
stage2_model = FaceKeypointModelStage2().to(config.DEVICE)
# optimizer
optimizer = optim.Adam(stage1_model.parameters(), lr=config.LR)
# we need a loss function which is good for regression like MSELoss
criterion = nn.MSELoss()

# training function
def fit(model, dataloader, data):
    print('Training')
    stage1_model.train()
    stage2_model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    print("Number of batches: ", num_batches)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        print("STAGE 1")
        print("--------------------------------------------------------------------------------")
        counter += 1
        image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
        # print("Input: ", image.shape)
        # print("Keypoints: ", keypoints)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        outputs = stage1_model(image)
        print("Output Training Shape: ", outputs.shape)
        # print("Output: ", outputs)
        #cropped_imgs = []
        list_of_all_window_shifts = []
        for keypoint in keypoints: #12
            list_of_window_shifts = []
            for index, element in enumerate(keypoint):
                if index % 2 == 0:
                    left = int(keypoint[index])-int(10/2)
                    top = int(keypoint[index+1])-int(10/2)
                    list_of_window_shifts.append((left,top))
            list_of_all_window_shifts.append(list_of_window_shifts)
        #print("List of Window Shift Elements: ", list_of_all_window_shifts)
        list_of_all_cropped_imgs = [] 
        for idx, img in enumerate(image): #12
            cropped_imgs = []
            for j in range(len(list_of_all_window_shifts[idx])): #8
                left = list_of_all_window_shifts[idx][j][0]
                top = list_of_all_window_shifts[idx][j][1]
                cropped_img = TF.crop(img=img, top=top, left=left, height=10, width=10)
                cropped_imgs.append(cropped_img)
            cropped_imgs = torch.cat(cropped_imgs, 0)
            print("Cropped Images: ", cropped_imgs.shape)
            list_of_all_cropped_imgs.append(cropped_imgs)
            #print("List of All Cropped Images: ", list_of_all_cropped_imgs)
        # print("List of All Cropped Images: ", list_of_all_cropped_imgs) #12 x 8
        #list_of_all_cropped_imgs = torch.cat(list_of_all_cropped_imgs, 0)
        #print("Cropped Images Shape2: ", list_of_all_cropped_imgs.shape) 
        #list_of_all_cropped_imgs = torch.reshape(list_of_all_cropped_imgs, (len(image)*8,1,10,10))
        #print("Cropped Images: ", list_of_all_cropped_imgs)
        #print("Cropped Images Shape: ", list_of_all_cropped_imgs.shape) #96,1,10,10
        # for keypoint in keypoints: #12
        #     for i in range(len(keypoint)): #16
        #         #print("Keypoint: ", int(keypoint[i]))
        #         top = int(keypoint[i])-int(10/2)
        #         left = int(keypoint[i])-int(10/2)
        #         cropped_img = TF.crop(img=image, top=top, left=left, height=10, width=10)
        #         cropped_imgs.append(cropped_img)
        # print("Cropped Images Shape: ", len(cropped_imgs)) #192
        print("STAGE 2")
        print("--------------------------------------------------------------------------------")
        outputs_stage2 = []
        count = 0
        # outputs2 = stage2_model(list_of_all_cropped_imgs)
        # print("Output2 Training Shape: ", outputs2.shape)
        for i in range(len(list_of_all_cropped_imgs)): #12
            #print("Imgs: ", list_of_all_cropped_imgs[i].shape)
            imgs = torch.reshape(list_of_all_cropped_imgs[i], (len(list_of_all_cropped_imgs[i]),1,10,10))
            print("Imgs: ", imgs.shape)
            outputs2 = stage2_model(imgs)
            print("Output2: ", outputs2)
            print("Output2 Training Shape: ", outputs2.shape)
            outputs2_avg = torch.mean(outputs2, dim=0)
            print("Output 2 Average: ", outputs2_avg)
            outputs_stage2.append(outputs2_avg)
        print("Output2 Stage 2 Shape: ", outputs_stage2)
        outputs_stage2 = torch.cat(outputs_stage2, 0)
        outputs_stage2 = torch.reshape(outputs_stage2, (12,16))
        print("Output2 Stage 2 Shape2: ", outputs_stage2.shape)
        # for img in list_of_all_cropped_imgs:
        #     count+=1
        #     print("Cropped Image: ", img.shape)
        #     outputs2 = stage2_model(img)
        #     outputs_stage2.append(outputs2)
        #     print("Output2 Training Shape: ", outputs2.shape)
        #     if(count == 191):
        #         print("Output2: ", outputs2)
        # print("Output stage 2: ", len(outputs_stage2))

        # loss = criterion(outputs, keypoints)
        # train_running_loss += loss.item()
        # loss.backward()
        # optimizer.step()
        loss = criterion(outputs_stage2, keypoints)
        train_running_loss += loss.item()
        loss.backward()
        optimizer.step()

        # Feed into another CNN
        # crop initial image 
        # cropped_imgs = []
        # for keypoint in keypoints:
        #     print("Keypoint: ", keypoint)
        #     cropped_img = TF.crop(img=image, top=keypoint-10/2, left=keypoint-10/2, height=10, width=10)
        #     cropped_imgs.append(cropped_img)
        # print("Cropped Images Shape: ", cropped_imgs.shape)

    train_loss = train_running_loss/counter
    return train_loss

# validatioon function
def validate(model, dataloader, data, epoch):
    print('Validating')
    stage1_model.eval()
    stage2_model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            print("STAGE 1")
            print("--------------------------------------------------------------------------------")
            counter += 1
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            # print("Input: ", image.shape)
            # print("Keypoints: ", keypoints)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = stage1_model(image)
            print("Output Training Shape: ", outputs.shape)
            # print("Output: ", outputs)
            #cropped_imgs = []
            list_of_all_window_shifts = []
            for keypoint in keypoints: #12
                list_of_window_shifts = []
                for index, element in enumerate(keypoint):
                    if index % 2 == 0:
                        left = int(keypoint[index])-int(10/2)
                        top = int(keypoint[index+1])-int(10/2)
                        list_of_window_shifts.append((left,top))
                list_of_all_window_shifts.append(list_of_window_shifts)
            #print("List of Window Shift Elements: ", list_of_all_window_shifts)
            list_of_all_cropped_imgs = [] 
            for idx, img in enumerate(image): #12
                cropped_imgs = []
                for j in range(len(list_of_all_window_shifts[idx])): #8
                    left = list_of_all_window_shifts[idx][j][0]
                    top = list_of_all_window_shifts[idx][j][1]
                    cropped_img = TF.crop(img=img, top=top, left=left, height=10, width=10)
                    cropped_imgs.append(cropped_img)
                cropped_imgs = torch.cat(cropped_imgs, 0)
                print("Cropped Images: ", cropped_imgs.shape)
                list_of_all_cropped_imgs.append(cropped_imgs)
                #print("List of All Cropped Images: ", list_of_all_cropped_imgs)
            # print("List of All Cropped Images: ", list_of_all_cropped_imgs) #12 x 8
            #list_of_all_cropped_imgs = torch.cat(list_of_all_cropped_imgs, 0)
            #print("Cropped Images Shape2: ", list_of_all_cropped_imgs.shape) 
            #list_of_all_cropped_imgs = torch.reshape(list_of_all_cropped_imgs, (len(image)*8,1,10,10))
            #print("Cropped Images: ", list_of_all_cropped_imgs)
            #print("Cropped Images Shape: ", list_of_all_cropped_imgs.shape) #96,1,10,10
            # for keypoint in keypoints: #12
            #     for i in range(len(keypoint)): #16
            #         #print("Keypoint: ", int(keypoint[i]))
            #         top = int(keypoint[i])-int(10/2)
            #         left = int(keypoint[i])-int(10/2)
            #         cropped_img = TF.crop(img=image, top=top, left=left, height=10, width=10)
            #         cropped_imgs.append(cropped_img)
            # print("Cropped Images Shape: ", len(cropped_imgs)) #192
            print("STAGE 2")
            print("--------------------------------------------------------------------------------")
            outputs_stage2 = []
            count = 0
            # outputs2 = stage2_model(list_of_all_cropped_imgs)
            # print("Output2 Training Shape: ", outputs2.shape)
            for i in range(len(list_of_all_cropped_imgs)): #12
                #print("Imgs: ", list_of_all_cropped_imgs[i].shape)
                imgs = torch.reshape(list_of_all_cropped_imgs[i], (len(list_of_all_cropped_imgs[i]),1,10,10))
                print("Imgs: ", imgs.shape)
                outputs2 = stage2_model(imgs)
                print("Output2: ", outputs2)
                print("Output2 Training Shape: ", outputs2.shape)
                outputs2_avg = torch.mean(outputs2, dim=0)
                print("Output 2 Average: ", outputs2_avg)
                outputs_stage2.append(outputs2_avg)
            print("Output2 Stage 2 Shape: ", outputs_stage2)
            outputs_stage2 = torch.cat(outputs_stage2, 0)
            outputs_stage2 = torch.reshape(outputs_stage2, (2,16))
            print("Output2 Stage 2 Shape2: ", outputs_stage2.shape)
            # for img in list_of_all_cropped_imgs:
            #     count+=1
            #     print("Cropped Image: ", img.shape)
            #     outputs2 = stage2_model(img)
            #     outputs_stage2.append(outputs2)
            #     print("Output2 Training Shape: ", outputs2.shape)
            #     if(count == 191):
            #         print("Output2: ", outputs2)
            # print("Output stage 2: ", len(outputs_stage2))

            # loss = criterion(outputs, keypoints)
            # train_running_loss += loss.item()
            # loss.backward()
            # optimizer.step()
            loss = criterion(outputs_stage2, keypoints)
            valid_running_loss += loss.item()
            if (epoch+1) % 25 == 0 and i == 0:
                utils.valid_keypoints_plot(image, outputs_stage2, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
    return valid_loss

# Start the training
train_loss = []
val_loss = []
for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1} of {config.EPOCHS}")
    train_epoch_loss = fit(stage1_model, train_loader, train_data)
    val_epoch_loss = validate(stage1_model, valid_loader, valid_data, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{config.OUTPUT_PATH}/loss.png")
plt.show()
torch.save({
            'epoch': config.EPOCHS,
            'model_state_dict': stage1_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{config.OUTPUT_PATH}/model.pth")
print('DONE TRAINING')