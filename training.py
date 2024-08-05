import os
import sys
import argparse
import os
import math
import pickle as pickle
import glob
import copy
import json
import logging
import time
import pandas as pd
from typing import Any, Callable, Optional, List, Sequence

import multiprocessing
from multiprocessing import Process
from subprocess import call

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import numpy as np
from scipy.io.wavfile import read

import torch
import torch.utils.data
from torch import nn, Tensor
from torch.nn import functional as F
from torchaudio import transforms as T
from torchaudio.prototype import transforms as PT
import torch.optim as optim

import torchaudio
import librosa

import pt_util


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from model import ResNet,resnet50


parser = argparse.ArgumentParser()
parser.add_argument("-d","--data_path", help = "Path to model Inputs(pkl files)")
parser.add_argument("-l","--log_path",help="Path to save the logs and model")
args = parser.parse_args()

MAX_LENGTH = 441000
num_channels = 2
# batch = 60


start_epoch = 0
# feature_path = '/blue/srampazzi/bhattr/music_genre_classification/labeled_snip_dataset/mfcc_features/'
# BASE_PATH = feature_path
# train_dir = os.path.join(feature_path,'train')
# test_dir = os.path.join(feature_path,'test')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

def get_mfcc(waveform, sample_rate):
  n_fft = 2048
  win_length = None
  hop_length = 512
  n_mels = 256
  n_mfcc = 256

  mfcc_transform = T.MFCC(
      sample_rate=sample_rate,
      n_mfcc=n_mfcc,
      melkwargs={
          "n_fft": n_fft,
          "n_mels": n_mels,
          "hop_length": hop_length,
          "mel_scale": "htk",
      },
  )

  return mfcc_transform(waveform)

def get_chromagram(waveform, sample_rate):
    chromaspectrogram = PT.ChromaSpectrogram(sample_rate=sample_rate,n_chroma=20, n_fft=1024)
    signal = chromaspectrogram(waveform)
    return signal

def get_specgram(waveform, n_fft=4000, win_length=4000,hop_length=2000,target_size=(224,224) ):
    # Ensure waveform is stereo
    if waveform.shape[0] != 2:
        raise ValueError("Audio file is not stereo.")

    # Create spectrogram transform
    spectrogram_transform = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)

    specgrams = []
    for i in range(waveform.shape[0]):  # Loop over channels (waveform is [channels, time])
        channel_spectrogram = spectrogram_transform(waveform[i].unsqueeze(0))  # Add channel dimension for transform
        
        # Resize the spectrogram to 224x224 using bilinear interpolation
        resized_spectrogram = F.interpolate(
            channel_spectrogram.unsqueeze(0),  # Add a batch dimension (B x C x H x W)
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)  # Remove the batch dimension
        
        specgrams.append(resized_spectrogram.squeeze(0))  # Remove channel dimension added for transform 
    # Stack spectrograms along the new channel dimension to maintain original channel separation
    stacked_spectrograms = torch.stack(specgrams)
   
    return stacked_spectrograms

    
class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, feature='raw_waveform',split='train',is_pickle=True,bins=140):
        super(MusicDataset, self).__init__()
        self.is_pickle = is_pickle
        self.data_dir = os.path.join(data_dir,feature)
        if is_pickle == True:
            self.data_pkl = glob.glob(os.path.join(self.data_dir,'*.pkl'))
            self.data = []
            for fpath in self.data_pkl:
                if split in fpath:
                    pkl_fd = open(fpath,'rb')
                    self.data = pickle.load(pkl_fd)
        else:
            self.data_dir = os.path.join(self.data_dir,split)
            # print(self.data_dir)
            self.data = glob.glob(os.path.join(self.data_dir,'*.mp3'))
            print(self.data_dir)
            # print(self.data)
        self.bins = bins
        # print(len(self.data))
        
        self.feature = feature

    def __len__(self):
        return len(self.data)

    def _get_feature_data(self, file_path, feature):
        song,sr = torchaudio.load(file_path,normalize=True)
        if feature == 'raw_waveform':
            # print(song.shape)
            return song
        if feature == 'mfcc':
            mfcc = get_mfcc(song,sr)
            # mfcc = mfcc[0,:,:]
            # print(mfcc.shape)
            return mfcc
        if feature == 'specgram':
            return get_specgram(song)
        if feature == 'chroma':
            chroma = get_chromagram(song, sr)
            # print(chroma)
            chroma = chroma[0,:,:]
            # print(chroma.shape)
            return chroma

    def __getitem__(self, index):
        self.timeseries_length = 1290
        feature_data = np.zeros(
            (self.timeseries_length, 33), dtype=np.float64
        )
        freq_bins = self.bins
        if self.is_pickle == True:
            model_data = self.data[index]
            # print("data shape:",model_data[0].shape)
            # print(model_data[0])
            spec1 = get_specgram(model_data[0],n_fft=4000, win_length=4000, 
                                                            hop_length=2000).numpy()

            spec2 = get_specgram(model_data[0],n_fft=2000, win_length=2000, 
                                                            hop_length=1000).numpy()

            spec3 = get_specgram(model_data[0],n_fft=1000, win_length=1000, 
                                                            hop_length=500).numpy()

            model_data[0] = torch.tensor(np.concatenate([spec1,
                                                           spec2,
                                                           spec3], axis=0))
            # print(
                # "Data shape:",model_data[0].shape)
        # print(model_data)
        else:
            model_data = self.data[index]
            feature_data = self._get_feature_data( model_data, 'raw_waveform')
            spec1 = get_specgram(feature_data,n_fft=4000, win_length=4000, 
                                                            hop_length=2000).numpy()

            spec2 = get_specgram(feature_data,n_fft=2000, win_length=2000, 
                                                            hop_length=1000).numpy()

            spec3 = get_specgram(feature_data,n_fft=1000, win_length=1000, 
                                                            hop_length=500).numpy()
            
            feature_data = torch.tensor(np.concatenate([spec1,spec2,spec3], axis=0))
            
            label = int(os.path.basename(model_data).split('.')[0].split('__')[-1])
            model_data = [feature_data, (torch.tensor([label]), model_data)]
    
        return model_data

def setup_logger(name_logfile, path_logfile):
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    fileHandler = logging.FileHandler(path_logfile, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger

def extract_song_split(filename):
    # parts = filename.split('/')[-1].split('.')[0]
    # print(filename)
    split_song = filename.split('__')[0]  # Assuming filename structure is consistent
    # print(split_song)
    # split, song = split_song.split('_')
    if 'train' in split_song:
        split = 'train'
    if 'test' in split_song:
        split = 'test'
    if 'valid' in split_song:
        split = 'valid'
    return pd.Series([split, split_song])

def print_song_accuracy(df,bins):
    # Apply the function to the DataFrame
    df[['split', 'song_name']] = df['file_names'].apply(extract_song_split)
    
    # Group by song_name and calculate the majority prediction for each song
    song_data = df.groupby('song_name').agg(
        predicted_song_label=pd.NamedAgg(column='snip_predict_label', aggfunc=lambda x: 1 if x.mean() > 0.5 else 0),
        true_song_label=pd.NamedAgg(column='snip_true_label', aggfunc=lambda x: 1 if x.mean() > 0.5 else 0)
    ).reset_index()
    
    # Calculate song-level accuracy
    correct_predictions = (song_data['predicted_song_label'] == song_data['true_song_label']).sum()
    total_songs = song_data.shape[0]
    song_accuracy = correct_predictions / total_songs

    print(f"{bins}_Song-Level Accuracy: {song_accuracy * 100:.2f}")
    
def train_binary(model, epoch, data_loader, device, optimizer,pbar, pbar_update, BATCH_SIZE,split,bins,method='train'):
    model.to(device)
    model.train()
    losses = []
    all_pred = []
    all_proba = []
    all_target = []
    all_file_names = []
    df = pd.DataFrame()
    correct = 0
    for batch_idx, (data, (target,file_names)) in enumerate(data_loader):
        # print(data, target)
        print(f"Data shape: {data.shape}, Target shape: {target.shape}")
        # bs, c, h, w = data.shape
        # data = data.reshape(bs, c, h*w)
        # print(data.shape)
        # data = torch.cat(data, dim=0)
        # target = torch.cat(target, dim=0)
        # target = target.squeeze()
        data = data.to(device)
        target = target.float()
        target = target.to(device)
        output = model(data)

        loss = model.loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pred = (F.sigmoid(output) > 0.5).float()
        correct_mask = pred.eq(target.view_as(pred))
        num_correct = correct_mask.sum().item()
        correct += num_correct
        pred_cpu = pred.cpu().numpy().flatten()
        target_cpu = target.cpu().numpy().flatten()
        all_target += target_cpu.tolist()
        all_proba += torch.sigmoid(output.detach().cpu()).numpy().flatten().tolist()
        all_pred += pred_cpu.tolist()
        all_file_names += file_names

        if batch_idx % 20:
            print(f'Train batch Accuracy: {num_correct}/{BATCH_SIZE}')
            print(f"Train batch: {epoch} [{(batch_idx+1) * BATCH_SIZE}/{len(data_loader) * BATCH_SIZE} ({100. * (batch_idx+1) / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        pbar.update(pbar_update)
    print(f'Train Epoch {epoch} Accuracy: {correct}/{len(data_loader.dataset)}')
    ave_losses = np.mean(losses)
    data = {
    'file_names': all_file_names,
    'snip_predict_label': all_pred,
    'snip_true_label': all_target,
    'snip_probability': all_proba
    }

    df = pd.DataFrame(data)
    print_song_accuracy(df,bins)
    # plot_trainloss(len(data_loader),losses)
    
    # df['epoch'] = [epoch] * BATCH_SIZE * len(losses)
    # df['batch_size'] = [BATCH_SIZE] * BATCH_SIZE * len(losses)
    # df['file_names'] = all_file_names
    # df['snip_predict_label'] = all_pred
    # df['snip_true_label'] = all_target
    # df['snip_probability'] = all_proba
    
    
    # df['losses'] = losses
    update_classification_json(f'classification_{method}_{bins}.json', df,batch_loss=losses,epoch=epoch,BATCH_SIZE=BATCH_SIZE)
    print(f"Train Epoch: {epoch} total average loss: {ave_losses:.6f}")
    return ave_losses

def update_classification_json(filepath, df, batch_loss, epoch, BATCH_SIZE, conf_matrix=None):
    # Check if the JSON file exists
    if os.path.exists(filepath):
        # Load the existing data into a list
        with open(filepath, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                # File is empty, start with an empty list
                data = []
    else:
        # If the file does not exist, start with an empty list
        data = []
    
    # Convert DataFrame to JSON, ensuring it is in a format that can be appended to a list
    df_json = df.to_dict(orient='records')
    
    # Append new data to the list
    data.append({
        "epoch": epoch,
        "batch_loss": batch_loss,
        "batch_size": BATCH_SIZE,
        "data": df_json
    })
    
    # Save back to JSON file
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def test_binary(model, epoch, data_loader, device, BATCH_SIZE,split,bins,method='train'):
    model.to(device)
    model.eval()
    correct = 0
    losses = []
    all_pred = []
    all_proba = []
    all_target = []
    all_mis_classified = []
    all_correctly_classified = []
    test_loss = 0
    df = pd.DataFrame()
    all_song_names = []
    all_snip_names = []
    all_file_names = []
    with torch.no_grad():
        for batch_idx, (data, (target,file_names)) in enumerate(data_loader):
            target = target.float()
            # bs, c, h, w = data.shape
            # data = data.reshape(bs, c, h*w)
            # data = torch.cat(data, dim=0)
            # target = torch.cat(target, dim=0)
            # target = target.squeeze()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            # print(target)
            # print(output)
            losses.append(model.loss(output, target).item())
            pred = (F.sigmoid(output) > 0.5).float()
            # print(pred)
            correct_mask = pred.eq(target.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            pred = (F.sigmoid(output) > 0.5).float()
            pred_cpu = pred.cpu().numpy().flatten()
            target_cpu = target.cpu().numpy().flatten()
            all_target += target_cpu.tolist()
            all_proba += torch.sigmoid(output.detach().cpu()).numpy().flatten().tolist()
            all_pred += pred_cpu.tolist()
            all_file_names += file_names
            
            # print("target: ", all_target)
            # print("prediction:",all_pred)
            # print("probabilities: ",all_proba)
            # print("song: ", all_song_names)
            # print("file names: ", all_file_names)
            if batch_idx % 20 == 0:
                print(f"Testing batch {batch_idx} of {len(data_loader)}")
                print(f"Correct count: {num_correct}")
    #epoch,batch size, split ,song name, snip name, snip probability, true label, train loss
    
    test_loss = np.mean(losses)
    # df['epoch'] = [epoch] * BATCH_SIZE * len(losses)
    # df['batch_size'] = [BATCH_SIZE] * BATCH_SIZE * len(losses)
    # df['file_names'] = all_file_names
    # df['snip_predict_label'] = all_pred
    # df['snip_true_label'] = all_target
    # df['snip_probability'] = all_proba
    # df['losses'] = losses
    data = {
    'file_names': all_file_names,
    'snip_predict_label': all_pred,
    'snip_true_label': all_target,
    'snip_probability': all_proba
    }

    df = pd.DataFrame(data)
    print_song_accuracy(df,bins)
    conf_matrix = confusion_matrix(all_target, all_pred)
    update_classification_json(f'classification_{method}_{bins}.json',df,batch_loss=losses,epoch=epoch,BATCH_SIZE=BATCH_SIZE,conf_matrix=conf_matrix)    
    print(f'Confusion matrix:\n{conf_matrix}')
    class_report = classification_report(all_target, all_pred)
    print(f'Classification Report:\n{class_report}')
    area_under_curv = 0.0
    if method != 'post_prog':
        area_under_curv = roc_auc_score(all_target, all_proba, multi_class='ovr')
        print(f'Area Under The Curve:\n{area_under_curv}')
    print(f"\n{bins}_Test Epoch: {epoch}\tAccuracy: {correct}/{len(data_loader.dataset)} ({100. * correct / (len(data_loader.dataset)):.2f}%). Average loss is: {test_loss:.6f}\n")
    test_accuracy = 100. * correct / len(data_loader.dataset)

    return test_loss, test_accuracy, area_under_curv

def plot_loss(n_epochs,loss,split,str,bins):
    epochs = [e[0] for e in loss]
    loss = [e[1] for e in loss]
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, label=f'{split} {str}')
    plt.title(f'{split} {str} Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(f'{str}')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(f'{bins}_{split}_{str}.png')
    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(BASE_PATH,feature,freq_bins,method='train'):
    BATCH_SIZE = 30
    learning_rate = 0.001
    num_epochs = 20
    WEIGHT_DECAY = 0.00005
    SCHEDULER_EPOCH_STEP = 4
    SCHEDULER_GAMMA = 0.8
    USE_CUDA = True
    PRINT_INTERVAL = 10
    VALID_BATCH_SIZE = 64
    num_workers = 20
    use_cuda = USE_CUDA and torch.cuda.is_available()
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    eval_area_under_curvs = []

    # model = TwoDCNN()
    # model = Deep1DCNN()
    model = resnet50(in_channels=6,n_classes=1)
    # model = PretrainedModelHuBERT(num_input_channels=2, num_classes=1, dropout=0.05, model_name="facebook/hubert-large-ls960-ft")
    # model = LSTM(input_dim=33, hidden_dim=128, batch_size=BATCH_SIZE, output_dim=1, num_layers=3)
    model.to(device)
    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}
    print(f"count parameters: {count_parameters(model)}")
    train_set = MusicDataset(BASE_PATH, feature,split='train',bins=freq_bins)
    test_set = MusicDataset(BASE_PATH, feature,split='test',bins=freq_bins)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,**kwargs)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,**kwargs)


    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                            weight_decay= WEIGHT_DECAY, amsgrad=True)
    start_epoch = 0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_EPOCH_STEP, gamma=SCHEDULER_GAMMA)

    pbar_update = 1 / (len(train_loader)+len(test_loader))
    prev_learning_rate = learning_rate
    with tqdm(total=num_epochs) as pbar:
        try:
            for epoch in range(start_epoch, num_epochs + 1):
                train_loss, eval_loss, eval_accuracy, eval_area_under_curv = 0, 0, 0, 0

                train_loss = train_binary(model, epoch, train_loader, device, optimizer, pbar, pbar_update, BATCH_SIZE,'train',bins=freq_bins,method='train')
                eval_loss, eval_accuracy, eval_area_under_curv = test_binary(model, epoch, test_loader, device, VALID_BATCH_SIZE,'test',bins=freq_bins,method='test')

                model.save_model( BASE_PATH + f'/checkpoints_{freq_bins}/{epoch}.pt',num_to_keep=num_epochs)
                scheduler.step()
                train_losses.append((epoch, train_loss))
                eval_losses.append((epoch, eval_loss))
                eval_accuracies.append((epoch, eval_accuracy))
                eval_area_under_curvs.append((epoch, eval_area_under_curv))
                print(train_losses)
                print(eval_accuracies)
            plot_loss(num_epochs,train_losses,'train',str='Loss',bins=freq_bins)
            print("Train Loss:",train_loss)
            plot_loss(num_epochs,eval_losses,'eval',str='Loss',bins=freq_bins)
            print("Eval Loss:",eval_loss)
            plot_loss(num_epochs,eval_accuracies,'eval accuracy',str='accuracy',bins=freq_bins)
            print("eval accuracy",eval_accuracies)
        except KeyboardInterrupt as ke:
            print('Interrupted')
        except:
            import traceback
            traceback.print_exc()
        finally:
            print('Saving final model')
            model.save_model(BASE_PATH + f'/checkpoints_{freq_bins}/f{epoch}.pt',num_to_keep=num_epochs)
            
            
            return model, device

def validate_a_classifier(BASE_PATH, device_instant, num_classes=1,feature='raw_waveform',is_pickle=False,split='valid',freq_bins=140,method='train',epoch=0):    
    num_workers = 1
    EPOCHS = 5000
    BATCH_SIZE=30
    USE_CUDA = True
    LOG_PATH = BASE_PATH + '/logs/log' + '1' + '.pkl'

    use_cuda = USE_CUDA and torch.cuda.is_available()

    device = device_instant
    print(f'Using device {device}')
    
    print(f'num workers: {num_workers}')

    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if use_cuda else {}
    # model = TwoDCNN()
    # model = Deep1DCNN()
    model = resnet50(in_channels=6,n_classes=1)
    # print(f"Updated_model({0}): {model}")
    # model = PretrainedModelHuBERT(num_input_channels=2, num_classes=1, dropout=0.05, model_name="facebook/hubert-large-ls960-ft")
    model.to(device)
    # print(f"{model}")
    
    valid_set = MusicDataset(BASE_PATH, feature,split=split,is_pickle=is_pickle,bins=freq_bins)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE,shuffle=True, drop_last=False,**kwargs)

    start_epoch=epoch
    start_epoch = model.load_model(f'/blue/srampazzi/bhattr/music_genre_classification/labeled_snip_dataset/checkpoints_{freq_bins}/{epoch}.pt')
    start_epoch=epoch
    print(f"count parameters: {count_parameters(model)}")
    print(f"Updated_model({start_epoch}): {model}")
    print(f'Validation results:')
    # train_losses, eval_losses, eval_accuracies, eval_area_under_curvs = pt_util.read_log(LOG_PATH, ([], [], [], []))
    eval_loss, eval_accuracy, eval_area_under_curv = test_binary(model, start_epoch, valid_loader, device, BATCH_SIZE,split,bins=freq_bins,method=method)

    return model, device



test_drop_path = '/blue/srampazzi/bhattr/music_genre_classification/labeled_test_snip/'
drop_path = '/blue/srampazzi/bhattr/music_genre_classification/labeled_snip_dataset/'
other_path = '/blue/srampazzi/bhattr/music_genre_classification/labeled_test_snip/'

# for f in [140,224,300,370,400]:
f= 140
# train(drop_path,feature='raw_waveform',freq_bins=f,method='train')
for e in [3,4,10,11,12,17,20]:
    # validate_a_classifier(drop_path, device,feature='raw_waveform',is_pickle=True,split='valid',freq_bins=f,method='valid',epoch=e)
    validate_a_classifier(test_drop_path, device,feature='raw_waveform',is_pickle=False,split='test',freq_bins=f,method='test_set',epoch=e)
    validate_a_classifier(test_drop_path, device,feature='raw_waveform',is_pickle=False,split='valid',freq_bins=f,method='post_prog',epoch=e)