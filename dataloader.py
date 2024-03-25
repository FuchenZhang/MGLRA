import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
import json

class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open('E:\MM-DFN-main\data\iemocap\iemocap_data.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        del self.videoVisual['Ses05F_script02_2']
        del self.videoAudio['Ses05F_script02_2']
        del self.videoSpeakers['Ses05F_script02_2']
        del self.videoLabels['Ses05F_script02_2']
        self.testVid.remove('Ses05F_script02_2')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in \
                                  self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in
                dat]





class MELDDataset(Dataset):

    def __init__(self, path, n_classes, train=True):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _,self.videodata, \
            self.videoAudio, self.videoSentence, self.trainVid, \
            self.testVid, self.videoLabels= pickle.load(open('./meld_data.pkl', 'rb'))
            self.videoIDs1, self.videoSpeakers1, self.data, self.videoAudio1, self.videoSentence1, self.trainVid1, self.testVid1, self.videoLabels1 = pickle.load(open('./MELD_features_raw1.pkl', 'rb'))

            del self.videoAudio[1432]
            del self.videoSpeakers[1432]
            del self.videoLabels[1432]
            self.testVid.remove(1432)


        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
            self.videoAudio, self.videoSentence, self.trainVid, \
            self.testVid, _ = pickle.load(open('./meld_data.pkl', 'rb'))
            self.videoIDs1, self.videoSpeakers1, self.data, self.videoAudio1, self.videoSentence1, self.trainVid1, self.testVid1, self.videoLabels1 = pickle.load(open('./MELD_features_raw1.pkl', 'rb'))
            # input = []
            del self.videoAudio[1432]
            del self.videoSpeakers[1432]
            del self.videoLabels[1432]
            self.testVid.remove(1432)
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        return torch.FloatTensor(self.data[vid]),torch.FloatTensor(self.videoAudio1[vid]),torch.FloatTensor(self.videoSpeakers[vid]),torch.FloatTensor([1] * len(self.videoLabels[vid])),torch.LongTensor(self.videoLabels[vid]),vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 3 else pad_sequence(dat[i], True) if i < 5 else dat[i].tolist() for i in
                dat]