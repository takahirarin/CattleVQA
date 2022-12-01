import torch
import torch.nn as nn
from transformers import ViltProcessor
from vilt_config import ViltConfig
from dataset import VQADataset, read_data, collate_fn
from transformers import ViltForQuestionAnswering
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from vilt_model import ViltModel
import wandb # 実験結果の可視化ツール


# load data
def load_set(question,annotation,config):
    
    questions, annotations = read_data(question,annotation,config)
    # processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    trainset = VQADataset( questions, annotations,config)

    # make trainloader
    dataloader = DataLoader(trainset, collate_fn=collate_fn,batch_size =config.batch_size,shuffle=True )
    return dataloader

def load_set_split(question,annotation,config):# データセット分割用のloader
    
    questions, annotations = read_data(question,annotation,config)
    dataset = VQADataset(questions, annotations,config)
    # train:val = 8:2に分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # print(f"full: {len(dataset)} -> train: {len(train_dataset)}, test: {len(val_dataset)}")

    # make trainloader
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn,batch_size =config.batch_size,shuffle=True )
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn,batch_size =config.batch_size,shuffle=True )

    return train_dataloader, val_dataloader
    

# train
def train(model,trainloader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(150):  # loop over the dataset multiple times
        print(f"Epoch: {epoch}")
        for batch in tqdm(trainloader):
            # if batch == False:
            #     print("skip {} batch".format(count))
            #     continue
            # get the inputs; 
            batch = {k:v.to(device) for k,v in batch.items()}

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            print("Loss:", loss.item())
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss})
            wandb.watch(model)
        
    model.save_pretrained('weights/vilt_finetune')

# valiation
def val(model,valloader):
    model.eval()
    train_correct = 0
    train_running_loss = 0
    train_total = 0
    for batch in tqdm(valloader):
            batch = {k:v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits #各クラスへの確率
            _label = logits.argmax(-1) #バッチ内のそれぞれに対して最大のもののインデックスを返す、予測データ
            label = batch['labels'].argmax(-1)#正解データ
            train_correct += _label.eq(label).sum() #予測ラベルと正解ラベルが一致した数だけ加算
            train_running_loss += outputs.loss.item()
            train_total += len(label)
            wandb.log("train loss",train_running_loss)
            print(f'train_correct: {train_correct}')
        
    train_acc = train_correct/train_total
    print(f'train_total: {train_total}')
    print(f'train_correct: {train_correct}')
    print(f'train_acc: {train_acc}')
    
    

if __name__ == '__main__':
    #load config
    config = ViltConfig()
    wandb.init(project="vilt", name='vqa_train')
    wandb.config = {"learning_rate": 5e-5,  "epochs": 50, "batch_size": 32 }
    #load train data
    # train_quenstion = 'Dataset/questions/v2_OpenEnded_mscoco_train2014_questions.json'
    # train_annotation = 'Dataset/annotations/v2_mscoco_train2014_annotations.json'
    # trainloader = load_set(train_quenstion, train_annotation)

    # load val data
    # valデータをtrainとvalに分割して使用する
    val_question = 'Dataset/questions/v2_OpenEnded_mscoco_val2014_questions.json'
    val_annotation = 'Dataset/annotations/v2_mscoco_val2014_annotations.json'

    train_dataloader, val_dataloader = load_set_split(val_question, val_annotation, config)
    
    # load model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """今後自分のモデルを使う予定"""
    # model = ViltModel()
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 num_labels=len(config.id2label),
                                                 id2label=config.id2label,
                                                 label2id=config.label2id)
    #model = ViltForQuestionAnswering.from_pretrained('weights/vilt_finetune')                         
    #print(model)
    model.to(device)
    train(model,train_dataloader)
   # val(model,valloader)


    # fine-tuneも結構時間かかるので、検証のコードまたテストされていない
    # weights = 'weights/vqa_finetune.pth'
    # val(model,valloader,weights)


