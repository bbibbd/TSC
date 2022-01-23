import os
import time
import torch
import cv2

import numpy as np
import matplotlib.pyplot as plt


class Trainer(object):

    def __init__(self, epochs, model, optimizer, criterion, trainloader, validloader, scheduler, ckptroot):

        super(Trainer, self).__init__()

        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainloader = trainloader
        self.validloader = validloader
        self.scheduler = scheduler
        self.ckptroot = ckptroot
        self.output_list=[]


    def train(self):
        self.model.to('cuda')
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for(images, labels) in self.trainloader:
            images = images.cuda()
            labels = labels.cuda()
            self.optimizer.zero_grad()
            output, _=self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()
            acc = self.calculate_accuracy(output, labels)
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss/len(self.trainloader), epoch_acc/len(self.trainloader)

    def validate(self):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()

        with torch.no_grad():
            for(images, labels) in self.validloader:
                images = images.cuda()
                labels = labels.cuda()
                
                # Run predictions
                output, _ = self.model(images)
                loss = self.criterion(output, labels)
                
                # Calculate accuracy
                acc = self.calculate_accuracy(output, labels)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        
        return epoch_loss / len(self.validloader), epoch_acc / len(self.validloader)
    
    # def test(self):
    #     test_loss = 0
    #     self.model.eval()
    #     with torch.no_grad():
    #         for(images) in self.validloader:
    #             #images = images.cuda()
    #             #labels = labels.cuda()
                
    #             # Run predictions
    #             output, _ = self.model(images)
    #             loss = self.criterion(output)
    #             output = output.cpu.numpy()

    #             for out in output:
    #                 self.output_list.append(out[0])
    #             # Calculate accuracy
    #             #acc = self.calculate_accuracy(output, labels)
                
    #             test_loss += loss.item()
    #             #epoch_acc += acc.item()
        
    #     return test_loss / len(self.validloader)#, epoch_acc / len(self.validloader)            


    def perform_training(self):
        epochs = self.epochs

        train_loss_list = [0]*epochs
        train_acc_list = [0]*epochs
        val_loss_list = [0]*epochs
        val_acc_list = [0]*epochs
        
        for epoch in range(epochs):
            self.scheduler.step()
            print("==> Traning epoch %d..."%(epoch))
            train_start_time = time.monotonic()
            train_loss, train_acc = self.train()
            #train_loss = self.train()
            train_end_time = time.monotonic()

            val_start_time = time.monotonic()
            val_loss, val_acc = self.validate()
            #val_loss= self.validate()
            val_end_time = time.monotonic()

            train_loss_list[epoch] = train_loss
            train_acc_list[epoch] = train_acc
            val_loss_list[epoch] = val_loss
            val_acc_list[epoch] = val_acc
            
            print("Training: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (train_loss, train_acc, train_end_time - train_start_time))
            print("Validation: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" % (val_loss, val_acc, val_end_time - val_start_time))
            
            # print("Training: Loss = %.4f, Time = %.2f seconds" % (train_loss, train_end_time - train_start_time))
            # print("Validation: Loss = %.4f, Time = %.2f seconds" % (val_loss, val_end_time - val_start_time))
            
            print("")

            if epoch%3==0 or epoch ==self.epochs-1:
                print("==>Save Checkpoint ...")

                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }

                self.save_checkpoint(state)
        #plot loss graph
        train_loss_list = np.array(train_loss_list)
        validation_loss_list = np.array(val_loss_list)
        self.plot_train_loss(train_loss_list, validation_loss_list)
        #plot accuracy graph
        train_acc_list = np.array(train_acc_list)
        val_acc_list = np.array(val_acc_list)
        self.plot_accuracy(train_acc_list, val_acc_list)


    def save_checkpoint(self, state):
        """Save checkpoint."""
        if not os.path.exists(self.ckptroot):
            os.makedirs(self.ckptroot)

        torch.save(state, self.ckptroot + 'model-{}.h5'.format(state['epoch']))


    def plot_train_loss(self, train_loss_list, validation_loss_list):
        """"Plot Loss Graph."""
        plt.title("Training Loss vs validation Loss")
        plt.xlabel("Epcoh")
        plt.plot(range(len(train_loss_list)), train_loss_list, 'b', label='Training Loss')
        plt.plot(range(len(train_loss_list)), validation_loss_list, 'g', label='Validation Loss')
        #plt.plot(np.linspace(0, len(train_loss_list), len(validation_loss_list)), validation_loss_list, 'g-.', label='Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_accuracy(self, train_acc_list, val_acc_list):
        plt.title("Accuracy Train vs validation")
        plt.xlabel("Epoch")
        plt.plot(range(len(train_acc_list)), train_acc_list, 'b', label='Training')
        plt.plot(range(len(train_acc_list)), val_acc_list, 'g', label='Validation')
        #plt.plot(np.linspace(0, len(train_loss_list), len(validation_loss_list)), validation_loss_list, 'g-.', label='Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()        

    def calculate_accuracy(self, output, label):
        top_pred = output.argmax(1, keepdim = True)
        correct = top_pred.eq(label.view_as(top_pred)).sum()
        acc = correct.float() / label.shape[0]
        return acc


