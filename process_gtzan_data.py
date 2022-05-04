#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:05:40 2022
@author: Matt
"""
import os
import librosa
import math
import json
import warnings
import time
warnings.filterwarnings('ignore')
# DATASET_PATH = 
# JSON_PATH = 
SAMPLE_RATE = 22050
DURATION = 30 # measure in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
def save_mfcc(dataset_path,json_path,n_mfcc=13,n_fft=2048,hop_length=512,num_segments=5):
    k=0
    # dictionary to store data
    data = {
        "mapping":[],
        "mfcc":[],
        "mel":[],
        "labels":[]
        }
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) # can be a float, round number to higher integer 1.2 -> 2
    # loop through all the genres
    for i,(dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):
        # ensure that we are not at the root level
        if dirpath is not dataset_path:
            # save semantic label
            dirpath_components = dirpath.split('/')  # genre_original/blues => [‘genre_original’,‘blues’]
            semantic_label = dirpath_components[-1]
            data['mapping'].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))
            # process files for a specific genre
            for f in filenames:
                #if k > 4: break
                k = k + 1
                print(k)
                # load audio file
                file_path = os.path.join(dirpath,f)
                signal,sr = librosa.load(file_path,sr = SAMPLE_RATE)
                # process segements extracting mfcc and storinng data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s # s==0 -> 0
                    finish_sample = start_sample + num_samples_per_segment # s==0 -> num_samples_per_segment
                    mfcc = librosa.feature.mfcc(signal[start_sample : finish_sample],
                                                sr = sr,
                                                n_fft = n_fft,
                                                n_mfcc = n_mfcc,
                                                hop_length = hop_length)
                    mfcc = mfcc.T
                    S = librosa.feature.melspectrogram(y=signal[start_sample : finish_sample], sr=sr)
                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data['mfcc'].append(mfcc.tolist())
                        data['labels'].append(i-1)
                        data['mel'].append(S.tolist())
                        #print(“{},segment:{}“.format(file_path,s))
    with open(json_path,"w") as fp:
        json.dump(data,fp, indent=4)
if __name__ == "__main__":
     save_mfcc(DATASET_PATH,JSON_PATH,num_segments=10)