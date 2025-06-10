from huggingface_hub import login
login(token="hf_")
model_number = "1"
model_name = "XLSR_dualpathfe_temp_"

repo_name=model_name+model_number

seed=34253
from transformers import set_seed
set_seed(seed)


# load the dataset
import datasets
from datasets import load_dataset # , load_metric
import evaluate



base_url_train = "/data/p312702/from_wietse/train_fy_nl/"
base_url_validation = "/data/p312702/from_wietse/validation_fy_nl/"
#base_url_test = "/data/p312702/from_wietse/test_fy_nl/"


cv_germanic_train = load_dataset("audiofolder",data_dir=base_url_train)
cv_germanic_validation=load_dataset("audiofolder",data_dir=base_url_validation)




def filter_function(example):
    # Example: filter rows where a specific column equals a certain value
    return example['lid'] in ['fy-NL','nl']



cv_frisian_nl_de_train = cv_germanic_train['train'].filter(filter_function)
cv_frisian_nl_de_validation = cv_germanic_validation["train"].filter(filter_function)



# show random elements
from datasets import ClassLabel
import random
import pandas as pd
import torchaudio
import re

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    df = pd.DataFrame(dataset[picks])
    #print(df)
    #display(HTML(df.to_html()))


# remove special characters (normalization)
import re

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', str(batch["transcription"])).lower()
    batch["transcription"] = "["+(batch["lid"].upper())+"]"+" "+batch["transcription"]
    #batch["lid_tokens"] = batch["lid"]
    return batch

cv_frisian_nl_de_train = cv_frisian_nl_de_train.map(remove_special_characters)
cv_frisian_nl_de_validation = cv_frisian_nl_de_validation.map(remove_special_characters)





def extract_all_chars(batch):
  all_text = " ".join(batch["transcription"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}




vocabs_train = cv_frisian_nl_de_train.map(extract_all_chars) # , batched=True, batch_size=-1, keep_in_memory=True,remove_columns=cv_germanic_train.column_names[0])
vocabs_validation=cv_frisian_nl_de_validation.map(extract_all_chars)


vocabs_train = vocabs_train.remove_columns(["transcription","lid","audio"])
vocabs_validation = vocabs_validation.remove_columns(["transcription","lid","audio"])


#print(vocabs_train["train"]["vocab"])

merged_list_train = []
merged_list_validation = []

for sublist in vocabs_train["vocab"]:
    merged_list_train.extend(sublist)

for sublist in vocabs_validation["vocab"]:
     merged_list_validation.extend(sublist)

#print(merged_list_train)

merged_list_train=[element for sublist in merged_list_train for element in sublist]
merged_list_validation=[element for sublist in merged_list_validation for element in sublist]

#print(merged_list_train)
#print(merged_list_validation)

vocab_list = list(set(merged_list_train) | set(merged_list_validation))


vocab_dict = {v: k for k, v in enumerate(vocab_list)}


vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
vocab_dict["[FY-NL]"] = len(vocab_dict)
vocab_dict["[NL]"] = len(vocab_dict)


print("length of vocab dict: ",len(vocab_dict))

#target_lang = "fry"
#new_vocab_dict = {target_lang: vocab_dict}
import json
with open('/data/p312702/from_wietse/'+model_name+model_number+'/vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)





from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("/data/p312702/from_wietse/"+model_name+model_number+"/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|") # target_lang=target_lang
repo_name = model_name+model_number
tokenizer.push_to_hub(repo_name)


#print("Tokenizer vocab size:", len(tokenizer))

from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False) #48000
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)




def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    # batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids

    #if batch["lid"] == "nl":
    #    batch["lid_tokens"] = 0
    #elif batch["lid"] == "fy-NL":
    #    batch["lid_tokens"] = 1 
    return batch



# This may take a long while
# save to disk

cv_frisian_nl_de_train=cv_frisian_nl_de_train.map(prepare_dataset)
cv_frisian_nl_de_validation=cv_frisian_nl_de_validation.map(prepare_dataset)
#cv_frisian_nl_de_test=cv_frisian_nl_de_test.map(prepare_dataset)


cv_frisian_nl_de_train=cv_frisian_nl_de_train.remove_columns(["transcription","audio","lid"])
cv_frisian_nl_de_validation=cv_frisian_nl_de_validation.remove_columns(["transcription","audio","lid"])
#cv_frisian_nl_de_test=cv_frisian_nl_de_test.remove_columns(["transcription","audio","lid"])





import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        #lid_features =  [feature["lid_tokens"] for feature in features]


        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        #batch["lid_tokens"] = torch.tensor(lid_features)

        return batch



data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


from evaluate import load
#from datasets import load_metric
#wer_metric = load_metric("wer")
wer_metric=evaluate.load("wer")

import numpy as np

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    for i in range(min(3, len(pred_str))):
        print(f"Prediction: {pred_str[i]}")
        print(f"Reference:  {label_str[i]}")
        print("-" * 40)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


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

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
import soundfile

import copy


'''
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


'''

class Wav2Vec2ForCTCDualFEAdd(Wav2Vec2ForCTC):
        def __init__(self, config, target_lang=None):
                super().__init__(config,target_lang)
                self.wav2vec2.feature_extractor=DualFEAdd(self.wav2vec2.feature_extractor)


'''

class Wav2Vec2ForCTCDualFEConv(Wav2Vec2ForCTC):
        def __init__(self, config, target_lang=None):
                super().__init__(config,target_lang)
                self.wav2vec2.feature_extractor=DualFEConv(config, self.wav2vec2.feature_extractor)
'''
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Config


#model = Wav2Vec2ForCTCDualFEConv.from_pretrained(


#config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-1b")
#config.update({
#"vocab_size": len(processor.tokenizer),
#"final_dropout": 0.1,
#"hidden_dropout":0.1,
#"intermediate_size": 5120,
#})

from transformers import Wav2Vec2Model


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MoLE model
#model = Wav2Vec2ForCTCMoLE(config)
#model = model.to(device)
#print("model device: ", device)
#model.wav2vec2.load_state_dict(
#	Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xls-r-1b").wav2vec2.state_dict()
#)




#     ORIGINAL:
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-1b",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer), #adapter)attn_dim
    ignore_mismatched_sizes=True,
)




#model.target_lang="fry"

finetune ="encoder"
if finetune == "adapter":
    model.init_adapter_layers()
    model.wav2vec2.feature_extractor=DualFEAdd(model.wav2vec2.feature_extractor)
    #model.wav2vec2.feature_extractor=DualFEConv(model.config, model.wav2vec2.feature_extractor)

    #model.freeze_base_model()
    for param in model.wav2vec2.encoder.parameters(): # model.wav2vec2.encoder.parameters()
        param.requires_grad = False

    adapter_weights = model._get_adapters()
    for param in adapter_weights.values():
        param.requires_grad = True

elif finetune == "encoder":
    pass
    #model.freeze_feature_extractor()
    model.wav2vec2.feature_extractor = DualFEAdd(model.wav2vec2.feature_extractor)
    #model.wav2vec2.feature_extractor=DualFEConv(model.config, model.wav2vec2.feature_extractor)


'''
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-1b",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    ignore_mismatched_sizes=True,
    adapter_attn_dim=16,
    vocab_size=len(tokenizer), #,150
)


print("model config vocab size: ",model.config.vocab_size)
print("Tokenizer length:", len(tokenizer))

model.target_lang = "fry"
'''
#model.freeze_feature_encoder()


#model.wav2vec2.feature_extractor = DualFEAdd(model.wav2vec2.feature_extractor)
#model.wav2vec2.feature_extractor = DualFEConv(model.config, model.wav2vec2.feature_extractor)
print(model)

#model.target_lang = "fry"
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2AttnAdapterLayer


print("model config vocab size: ",model.config.vocab_size)
print("Tokenizer length:", len(tokenizer))


'''
model.init_adapter_layers()
model.freeze_base_model()
adapter_weights = model._get_adapters()
for param in adapter_weights.values():
	param.requires_grad = True
'''


#model.resize_token_embeddings(len(tokenizer))
#model.init_adapter_layers()

#model.config.vocab_size = tokenizer.vocab_size
#for module in model.wav2vec2.modules():
	#if isinstance(module, Wav2Vec2AttnAdapterLayer):
	#	module.requires_grad=True
	#else:
#		module.requires_grad=False



from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir=repo_name,
  group_by_length=True,
  seed=seed,
  data_seed=seed,
  per_device_train_batch_size=4,
  per_device_eval_batch_size=4,
  gradient_accumulation_steps=32,
  eval_accumulation_steps=8,
  evaluation_strategy="steps",
  num_train_epochs=60,
  fp16=True,
  gradient_checkpointing=True,
  save_steps=100,
  save_strategy="no",
  eval_steps=100,
  logging_steps=100,
  learning_rate=1e-4,
  weight_decay=0.001,
  warmup_steps=1000,
  save_total_limit=2,
  adam_beta2=0.98,
  warmup_ratio=0.1,
)


# Here you can choose whether to use validated data only or use validated+invalidated
#common_voice_train_invalidated = datasets.concatenate_datasets([common_voice["train"],common_voice["invalidated"]])
#train_datasett = common_voice_train_invalidated 
train_dataset = cv_frisian_nl_de_train
eval_dataset = cv_frisian_nl_de_validation
#test_dataset=cv_frisian_nl_de_test['train']
#test_dataset = cv_germanic_test



#print(train_datasett["transcription"])

print("****************************************************************")
print("******************************start training*********************")
from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset, # common_voice["train]
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)


trainer.train()

trainer.save_model(repo_name)

