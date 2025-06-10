from huggingface_hub import login
login(token="hf_")

from datasets import Audio
import evaluate
#model_number = "36"



model_id="XLSR_dualpathfe_temp_1"

# /data/wav2vec2_frisian_common_voice_14
#__________________________________________________________________________________________________________
#seed=668912
#from transformers import set_seed
#set_seed(seed)
#________________________________________________________________________________________________________
# load the dataset
import datasets
from datasets import load_dataset # load_metric
#common_voice = load_dataset("fsicoli/common_voice_17_0", "fy-NL",use_auth_token=True, trust_remote_code=True )
#common_voice = common_voice.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))
#common_voice = common_voice.remove_columns(['client_id', 'up_votes','down_votes', 'age', 'gender','accent','locale','segment'])


#base_url_test="/data/common_voice_17_germanic/test_iranian/"
#base_url_test="/data/common_voice_17_germanic/test_romance"
base_url_test="/data/common_voice_17_germanic/test_fy_fr"


cv_germanic_test=load_dataset("audiofolder",data_dir=base_url_test,trust_remote_code=True)
cv_frisian_test=cv_germanic_test

def filter_function(example):
    # Example: filter rows where a specific column equals a certain value
    #return example['lid'] in ['fy-NL','nl','de']
    return example['lid'] == ('fy-NL')

cv_frisian_test=cv_frisian_test.filter(filter_function)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(cv_frisian_test)

cv_frisian_test['train'] = cv_frisian_test['train'].cast_column("audio", Audio(sampling_rate=16000))


print(cv_frisian_test)
#______________________________________________________________________________________________________
# show random elements
from datasets import ClassLabel
import random
import pandas as pd
import torchaudio



import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
import soundfile

import copy


class DualFEAdd(nn.Module):
    """
    This class implements the Dual-FE-Add feature extraction module of https://www.isca-archive.org/interspeech_2024/shi24b_interspeech.pdf
    """

    def __init__(self, feature_extractor):
        super().__init__()

        # this is the frozen feature extractor
        self.frozen_feature_extractor = copy.deepcopy(feature_extractor)
        for param in self.frozen_feature_extractor.parameters():
            param.requires_grad = False

        # this is the finetuned feature extractor
        self.finetuned_feature_extractor = feature_extractor
        for param in self.finetuned_feature_extractor.parameters():
            param.requires_grad = True

    def forward(self, input_values):
        hidden_states_frozen = hidden_states_finetuned = input_values[:, None]
        hidden_states_fused = None

        for conv_frozen, conv_finetuned in zip(
            self.frozen_feature_extractor.conv_layers,
            self.finetuned_feature_extractor.conv_layers,
        ):
            # input of frozen layer is the output of previous frozen layer
            hidden_states_frozen = conv_frozen(hidden_states_frozen)

            # input of finetuned layer is output of previous finetuned layer + previous fused output
            if hidden_states_fused is not None:
                hidden_states_finetuned += hidden_states_fused
            hidden_states_finetuned = conv_finetuned(hidden_states_finetuned)

            # fused output is output of frozen layer + output of finetuned layer
            hidden_states_fused = hidden_states_frozen + hidden_states_finetuned

        return hidden_states_fused




class DualFEConv(nn.Module):
    """
    This class implements the Dual-FE-Conv feature extraction module of https://www.isca-archive.org/interspeech_2024/shi24b_interspeech.pdf
    """

    def __init__(self, config, feature_extractor):
        super().__init__()

        # this is the frozen feature extractor
        self.frozen_feature_extractor = copy.deepcopy(feature_extractor)
        for param in self.frozen_feature_extractor.parameters():
            param.requires_grad = False

        # this is the finetuned feature extractor
        self.finetuned_feature_extractor = feature_extractor
        for param in self.finetuned_feature_extractor.parameters():
            param.requires_grad = True

        # these are the convolutional fusion layers
        # the frozen and finetuned feature extractors both consist of 7 convolutional layers that need to be fused
        # all have 512 output channels (see https://huggingface.co/facebook/wav2vec2-xls-r-1b/blob/main/config.json > "conv_dim")
        # therefore, the in_channel for each fusion layer is 2 * 512 = 1024 (since frozen and finetuned are concatenated) and the output should be 512
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(d * 2, d, 1) for d in config.conv_dim]
        )

    def forward(self, input_values):
        hidden_states_frozen = hidden_states_finetuned = input_values[:, None]
        hidden_states_fused = None

        for conv_frozen, conv_finetuned, conv_fusion in zip(
            self.frozen_feature_extractor.conv_layers,
            self.finetuned_feature_extractor.conv_layers,
            self.conv_layers,
        ):
            # input of frozen layer is the output of previous frozen layer
            hidden_states_frozen = conv_frozen(hidden_states_frozen)

            # input of finetuned layer is output of previous finetuned layer + output of previous fused layer
            if hidden_states_fused is not None:
                hidden_states_finetuned += hidden_states_fused
            hidden_states_finetuned = conv_finetuned(hidden_states_finetuned)

            # input of fused layer is concatenated output of frozen layer and output of finetuned layer
            hidden_states_fused = conv_fusion(
                torch.concat([hidden_states_frozen, hidden_states_finetuned], dim=1)
            )

        return hidden_states_fused




class Wav2Vec2ForCTCDualFEAdd(Wav2Vec2ForCTC):
	def __init__(self, config, target_lang=None):
		super().__init__(config,target_lang)
		self.wav2vec2.feature_extractor=DualFEAdd(self.wav2vec2.feature_extractor)

class Wav2Vec2ForCTCDualFEConv(Wav2Vec2ForCTC):
        def __init__(self, config, target_lang=None):
                super().__init__(config,target_lang)
                self.wav2vec2.feature_extractor=DualFEConv(self.config,self.wav2vec2.feature_extractor)


import re

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', str(batch["transcription"])).lower()
    batch["transcription"] = batch["transcription"]
    return batch

#cv_frisian_nl_de_train = cv_frisian_nl_de_train.map(remove_special_characters)
#cv_frisian_nl_de_validation = cv_frisian_nl_de_validation.map(remove_special_characters)
cv_frisian_test=cv_frisian_test.map(remove_special_characters)


from transformers import Wav2Vec2CTCTokenizer

#tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("/data/"+model_id, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
#tokenizer = Wav2Vec2CTCTokenizer('/data/vocab.json', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('/data/'+model_id, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
#tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('/data/'+model_id, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")


from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor

#feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False) #48000
#processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

'''
def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch


# This may take a long while
cv_frisian_test = cv_frisian_test.map(prepare_dataset, remove_columns=cv_frisian_test.column_names["train"], num_proc=1)
'''

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union



#from datasets import load_metric
#wer_metric = load_metric("wer")

from evaluate import load
wer_metric=evaluate.load("wer")


import numpy as np
'''
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
'''


from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor



#feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-1b")
#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/mms-1b-all")
#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/mms-1b")


#processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
#processor = Wav2Vec2Processor.from_pretrained("/data/"+model_id) 
#model = Wav2Vec2ForCTC.from_pretrained("/data/"+model_id).cuda() #"/checkpoint-"+target_checkpoint).cuda() # "Reihaneh/wav2vec2_frisian_common_voice_"+model_number).cuda()

#from transformers import AutoFeatureExtractor

#feature_extractor = AutoFeatureExtractor.from_pretrained("/data/p312702/from_wietse/"+model_id,trust_remote_code=True)

#feature_extractor=model.wav2vec2.feature_extractor
from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/data/p312702/from_wietse/"+model_id)
#print(type(feature_extractor))



processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
#from transformers import Wav2Vec2Processor

#processor = Wav2Vec2Processor.from_pretrained("/data/wav2vec2_fy_common_voice_91")


#model = Wav2Vec2ForCTCDualFEConv.from_pretrained("/data/"+model_id).cuda()
#model = Wav2Vec2ForCTC.from_pretrained("/data/"+model_id).cuda()
model = Wav2Vec2ForCTCDualFEAdd.from_pretrained("/data/"+model_id).cuda() 



#model.wav2vec2.feature_extractor = DualFEAdd(model.wav2vec2.feature_extractor)

#model = model.from_pretrained("/data/"+model_id).cuda()

#model.to("cuda")

#print("Tokenizer vocab size:", tokenizer.vocab_size)
#print("Model embedding size:", model.config.vocab_size)

# Resize the model's embedding layer
#model.resize_token_embeddings(len(tokenizer))

# Verify the change
#print("Updated Model embedding size:", model.config.vocab_size)

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch


# This may take a long while
cv_frisian_test = cv_frisian_test.map(prepare_dataset, remove_columns=cv_frisian_test.column_names["train"], num_proc=1)



import numpy as np
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


language_ids=[]

processor= Wav2Vec2Processor.from_pretrained("/data/"+model_id) #"Reihaneh/wav2vec2_frisian_common_voice_"+model_number)

#processor= Wav2Vec2Processor.from_pretrained("/data/"+model_id) #"Reihaneh/wav2vec2_frisian_common_voice_"+model_number)
 

def map_to_result(batch):
  model.to("cuda") 
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  #print(pred_ids)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0].split("]")[-1]
  if "]" in processor.batch_decode(pred_ids)[0]:
    language_ids.append(processor.batch_decode(pred_ids)[0].split("]")[-2])
  else:
    language_ids.append("ASDAS")
  print(batch["pred_str"])
  batch["sentence"] = processor.decode(batch["labels"], group_tokens=False)
  print(batch["sentence"])
  return batch


'''
def map_to_result(batch):
  model.to("cuda")
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  #print(pred_ids)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0].split("]")[-1]
  print(batch["pred_str"])
  batch["sentence"] = processor.decode(batch["labels"], group_tokens=False)
  print(batch["sentence"])
  return batch
'''
#processor = Wav2Vec2Processor.from_pretrained("/data/"+model_id) #"Reihaneh/wav2vec2_frisian_common_voice_"+model_number)



#wer_metric = load_metric("wer")
wer_metric=load("wer")
results = cv_frisian_test.map(map_to_result, remove_columns=cv_frisian_test['train'].column_names)
print(results)



#results = frisian_test.map(map_to_result, remove_columns=frisian_test.column_names)


print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["train"]["pred_str"], references=results["train"]["sentence"])))

#print(language_ids)
print(len(language_ids))
# calculate language identification accuracy:
correct_predictions=0
for element in language_ids:
	if element=="[FY-NL":
		correct_predictions+=1


print("correct predictions: ",correct_predictions)
acc=(correct_predictions/len(language_ids))*100
print("language identification accuracy: ",acc)

#___________________________________________________________________________________________________
#show_random_elements(results.remove_columns(["speech", "sampling_rate"]))

