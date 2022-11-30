import json
from tqdm.notebook import tqdm
from vilt_config import ViltConfig
import torch
import torch.nn as nn
import re
from typing import Optional
from PIL import Image
from os import listdir
from os.path import isfile, join
from transformers import ViltProcessor
from transformers import CLIPProcessor, CLIPVisionModel

def id_from_filename(filename: str) ->Optional[int]:
    match = filename_re.fullmatch(filename)
    if match is None:
        return None
    return int(match.group(1))

filename_re = re.compile(r".*(\d{12})\.((jpg)|(png))")
img_root = 'Dataset/val2014'
file_names = [f for f in listdir(img_root) if isfile(join(img_root, f))]
filename_to_id = {img_root + "/" + file: id_from_filename(file) for file in file_names}
id_to_filename = {v:k for k,v in filename_to_id.items()}



def read_data( path_question, path_annotation,config):
    # read quention file
    q = open(path_question)
    f_quenstion = json.load(q)
    questions = f_quenstion['questions'] 
    # read annotation file
    a = open(path_annotation)
    f_annotation = json.load(a)
    print(f_annotation.keys())
    annotations = f_annotation['annotations']
    # preprocess for annotation
    for annotation in tqdm(annotations):
        answers = annotation['answers']
        answer_count = {}
        for answer in answers:
            answer_ = answer["answer"]
            answer_count[answer_] = answer_count.get(answer_, 0) + 1
        labels = []
        scores = []
        for answer in answer_count:
            if answer not in list(config.label2id.keys()):
                continue
            labels.append(config.label2id[answer])
            score = get_score(answer_count[answer])
            scores.append(score)
        annotation['labels'] = labels
        annotation['scores'] = scores
    return questions, annotations
        
def get_score(count: int) -> float:
    return min(1.0, count / 3)

def collate_fn(batch):
    processor =  ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # create padded pixel values and corresponding pixel mask
    encoding = processor.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
    
    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels)
    
    return batch


class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, questions, annotations,config):
        self.questions = questions
        self.annotations = annotations
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.len = len(config.id2label)
        self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", hidden_size = 768)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
            # get image + text
            annotation = self.annotations[idx]
            questions = self.questions[idx]
            image = Image.open(id_to_filename[annotation['image_id']])
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
           
            text = questions['question']
            
            encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
            clip_embedding = self.vision_processor(images=image, return_tensors="pt")
            encoding['pixel_values'] = clip_embedding["pixel_values"]
            #clipのprocessorのreturnはpixel_valueしかないのでここのみ変更
            # remove batch dimension
            for k,v in encoding.items():
                encoding[k] = v.squeeze()
            # add labels
            labels = annotation['labels']
            scores = annotation['scores']
        
            targets = torch.zeros(self.len)
            for label, score in zip(labels, scores):
                targets[label] = score
            encoding["labels"] = targets

            return encoding


    
if __name__ == '__main__':
    config = ViltConfig()
    path1 = 'Dataset/questions/v2_OpenEnded_mscoco_val2014_questions.json'
    path2 = 'Dataset/annotations/v2_mscoco_val2014_annotations.json'
    questions, annotations = read_data(path1,path2,config)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    #dataset = VQADataset( questions, annotations,processor,config)
    dataset = VQADataset( questions, annotations,config)
    print(dataset[0])
