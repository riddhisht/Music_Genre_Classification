import numpy as np
import pandas as pd
import librosa
import glob
import os
import matplotlib.pyplot as plt
import random
import pickle
# !pip install PyDub
# from pydub import AudioSegment
from sklearn import preprocessing

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torchaudio.prototype as PT

print(torch.__version__)
print(torchaudio.__version__)

import librosa
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MAX_LENGTH = 441000
def remove_files(folder_path):
  for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    # print("Removing...",file_path)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # This will delete the file or link
        elif os.path.isdir(file_path):
            # If you also want to remove directories, uncomment the next line
            # shutil.rmtree(file_path)
            pass  # Currently, it does nothing with directories
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

def train_validate_test_split(df, train_percent=.5, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]].reset_index(drop=True)
    validate = df.iloc[perm[train_end:validate_end]].reset_index(drop=True)
    test = df.iloc[perm[validate_end:]].reset_index(drop=True)
    return train, validate, test

def rename_songs(df, split='train'):
    file_names = list()
    for index, row in df.iterrows():
        file_names.append(split + '_song_'+str(index))
    # df = pd.DataFrame({row['file_name'],row['label']})
    df['file_name'] = file_names
    return df

def get_feature():
  pass

# Take 10sec snippets.
def take_ns_snippets(audio, snippet_duration=10, sr=44100):

    snippet_length_samples = snippet_duration * sr
    
    # Calculate the number of full snippets and the length of the last snippet
    num_full_snippets = audio.shape[1] // snippet_length_samples
    last_snippet_length = audio.shape[1] % snippet_length_samples
    
    # Initialize a list to hold all snippets
    snippets = []
    
    # Extract full snippets
    for i in range(num_full_snippets):
        start_sample = i * snippet_length_samples
        snippet = audio[:,start_sample:start_sample + snippet_length_samples]
        # print(snippet.shape)
        if snippet.size(-1) != MAX_LENGTH:
            print(snippet.shape)
            print("snippet not of same size")
            assert snippet.shape[1] == MAX_LENGTH
        snippets.append(snippet)
    
    # Handle the last snippet if there's a remainder
    if last_snippet_length > 0:
        last_snippet = audio[:,-last_snippet_length:]
        # Calculate padding
        total_padding = (snippet_length_samples - last_snippet_length)
        if audio.ndim == 2: 
            padded_snippet = np.pad(last_snippet, ((0, 0), (0,total_padding)), mode='constant', constant_values=0)
        else:
            padded_snippet = np.pad(last_snippet, (0,total_padding), mode='constant', constant_values=0)
        padded_snippet = torch.from_numpy(padded_snippet).float()
        if padded_snippet.size(-1) != MAX_LENGTH:
            print("snippet not of same size")
        snippets.append(padded_snippet)
    
    return snippets

def store_snipped_data(df, folder_path, split, features):
    # snip_df = pd.DataFrame()
    data_split_path = os.path.join(folder_path,split)
    remove_files(data_split_path)
    total_snippets = 0
    pbar = tqdm(total=len(df.index), desc="Processing")
    for index, row in tqdm(df.iterrows()):
        # print(row['file_name'],row['label'])
        song_name = split + '_song_' + str(index)
        # print('changed_name: '+ song_name)
        print("Song Name: ",row['file_name'])
        print("Label: ",row['label'])
        song, sr = torchaudio.load(row['file_name'])
        if song.shape[0] == 1:
            pbar.update(1)
            continue
        # print("SAMPLE_RATE: ",sr)
        snippets = take_ns_snippets(song)

        # print("sampling rate:",sr)
        total_snippets += len(snippets)
        print("Number of snippets:",total_snippets)
        for id, snip in tqdm(enumerate(snippets)):
            # Make all the snippets same size/Discard < 10sec snippets
            snip_song_name = os.path.basename(row['file_name']).split('.')[0] + '__snip_' + str(id) +'__'+ str(row['label']) + '.mp3'
            print(snip_song_name)
            # print(snip)
            # print("snippet Length:",snip.shape)
            file_path = os.path.join(data_split_path,snip_song_name)
            # print(file_path)
            # if snip.shape[0] == 2 and snip.shape[1] >= MAX_LENGTH:
            total_snippets += 1
            torchaudio.save(file_path, snip, sample_rate=sr, format='mp3')
            # Save snippets
        pbar.update(1)
    print(f"{split} Snippets: ", total_snippets)

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

def get_specgram(waveform,sr):
    spectrogram = T.Spectrogram(n_fft=512)
    return spectrogram(waveform)

def prepare_model_input(folder_path,split,feature, feature_drop_path,save=False):
    data_path = os.path.join(folder_path,split)
    songs = glob.glob(os.path.join(data_path,'*.mp3'))
    #  Convert snips to tensor, label them, label with snip name as well.[tensor,label,song_name]
    # Make a pickle file and save it in same folder path if save = True
    dataset = []
    print(len(songs))
    for s in songs:
        # print(s)
        tens_wave, sr = torchaudio.load(s,normalize=True)
        # file_name = s.split('.')[0].split('/')[-1]
        file_name = s
        label = int(file_name.split('__')[-1].split('.')[0])
        if feature == 'raw_waveform':
            tens_data = tens_wave
            # print(label,file_name)
        if feature == 'mfcc':
            tens_data = get_mfcc(tens_wave, sr)
        if feature == 'specgram':
            tens_data = get_specgram(tens_wave,sr)
        if feature == 'chroma':
            tens_data = get_chromagram(tens_wave,sr)
        
        dataset.append([tens_data, (torch.tensor([label]),
                                                    file_name)])
    random.shuffle(dataset)
    random.shuffle(dataset)
    random.shuffle(dataset)
    random.shuffle(dataset)
    random.shuffle(dataset)
    if save == True:
        path = os.path.join(feature_drop_path, feature + '_' + split + '.pkl')
        print(path)
        pickle.dump(dataset,open(path,'wb'))

    return dataset


###############################################################################################

# pwd = os.path.curdir
features = ['raw_waveform']#,'mfcc','cqcc','spectogram']
# features = ['mfcc']
pwd = os.path.abspath(os.path.curdir)
print(pwd)
project_path = pwd 
# project_path = os.path.join(pwd,'drive', 'MyDrive', 'Prog_Rock_Project') # for running on Colab
#dataset_dir_path = os.path.join(project_path,'Small_Dataset')

dataset_dir_path = os.path.join(project_path,'Dataset')
prog_rock_path = os.path.join(dataset_dir_path,'Progressive_Rock_Songs')
non_prog_rock_path = os.path.join(dataset_dir_path, 'Not_Progressive_Rock')
non_prog_rock_other_path = os.path.join(non_prog_rock_path,'Other_Songs')
non_prog_rock_pop_path = os.path.join(non_prog_rock_path,'Top_Of_The_Pops')

drop_path = os.path.join(project_path, 'labeled_snip_dataset')

# test path
# test_dataset_dir_path = os.path.join(project_path,'test_set')
# prog_rock_path = os.path.join(dataset_dir_path,'Progressive_Rock_Songs')
# non_prog_rock_other_path = os.path.join(dataset_dir_path, 'Not_Progressive_Rock')
# os.mkdir(drop_path)
os.makedirs(drop_path,exist_ok=True)


# Iterate over all feature and generate train,test,valid folder.
for feature in features:
    feature_path = os.path.join(drop_path,feature)
    os.makedirs(feature_path,exist_ok=True)
################### NEED TO delete files if we want to UPDATE dataset ##########################

for s in ['train','test','valid']:
    os.makedirs(os.path.join(feature_path,s),exist_ok=True)

feature_dataset_list = [x for x in os.listdir(drop_path)]
print(feature_dataset_list)

# Read all the songs
prog_rock_files_list = glob.glob(os.path.join(prog_rock_path,'*.mp3'))
non_prog_rock_other_files_list = glob.glob(os.path.join(non_prog_rock_other_path,'*.mp3'))
non_prog_rock_pop_files_list = glob.glob(os.path.join(non_prog_rock_pop_path,'*.mp3'))
non_prog_rock_files_list = non_prog_rock_other_files_list + non_prog_rock_pop_files_list

prog_rock_label = [1 for s in prog_rock_files_list]
non_prog_rock_label = [0 for s in non_prog_rock_files_list]
df = pd.DataFrame(columns=['file_name','label'])
df['file_name'] = prog_rock_files_list + non_prog_rock_files_list
df['label'] = prog_rock_label + non_prog_rock_label

print(df)
df = df.sample(frac=1).reset_index(drop=True)
train_split, valid_split, test_split = train_validate_test_split(df, train_percent=0.7,validate_percent=0.15,seed=None)



# #Snipiffy
print("Snippifyy.....")
df_train = train_split
df_test =  test_split
df_valid = valid_split

# Store snippets
# print(df_valid)
feature_path_list = [os.path.join(drop_path,f) for f in features]
for f in features:
    feature_path = os.path.join(drop_path,f)
    print(feature_path)
    print("Snip Train Data")
    snip_df_train = store_snipped_data(df_train, feature_path, split='train',features=f)
    print("Snip Valid Data")
    snip_df_valid = store_snipped_data(df_valid, feature_path,split='valid',features=f)
    print("Snip Test Data")
    snip_df_test = store_snipped_data(df_test, feature_path,split='test',features=f)


# Convert all the songs to model input
snip_path = os.path.join(drop_path,'raw_waveform')
feature_path = '/blue/srampazzi/bhattr/music_genre_classification/labeled_snip_dataset/'
for f in features:
    fpath = os.path.join(feature_path)
    for split in ['train','test','valid']:
        dataset = prepare_model_input(snip_path, split, f,os.path.join(fpath,f),save=True)

# snip_path = '/blue/srampazzi/bhattr/music_genre_classification/labeled_test_snip/raw_waveform/'
# fpath = '/blue/srampazzi/bhattr/music_genre_classification/labeled_test_snip/'
# dataset = prepare_model_input(snip_path,'test','raw_waveform',fpath,save=True)


