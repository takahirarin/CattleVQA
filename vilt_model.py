from __future__ import absolute_import
import torch
import torch.nn as nn
from text_preprocess import BERTEmbeddings
from img_preprocess import PatchEmbeddings
from vilt_config import ViltConfig
import copy
from transformers.modeling_utils import  ModuleUtilsMixin
from transformer_block import MultiHeadAttention

from transformers import ViltProcessor
from dataset import VQADataset, read_data

class ViltModel(nn.Module):
    def __init__(self, config: ViltConfig):
        """"config: 'ViltConfig' instance """
        super(ViltModel, self).__init__()
        self.embeddings = ViltEmbeddings(config)
        self.encoder = ViltEncoder(config)
        self.pooler = ViltPooler(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = ViltClassifer(config)

    def forward(self,input_ids, attention_mask, token_type_ids,
                    pixel_values, pixel_mask, image_token_type_idx=1):
        input_shape = input_ids.size()
        # get text info
        text_batch_size, seq_length = input_shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(((text_batch_size, seq_length)), device=device)
        # get image info
        image_batch_size =  pixel_values.shape[0] 
        if pixel_values is None:
            pixel_values = torch.ones((image_batch_size, self.config.image_size, self.config_image_size), device=device)
        
        # calculate embeddings
        embeddings, masks = self.embeddings(
            input_ids, attention_mask, token_type_ids,
            pixel_values, pixel_mask,image_token_type_idx )
        
        # input embeddings into encoder
        extended_attention_mask = ModuleUtilsMixin.get_extended_attention_mask(attention_mask, input_shape)
        encoder_output = self.encoder(embeddings, extended_attention_mask)
        sequence_output = encoder_output[-1]
        pooled_output = self.pooler(sequence_output)

        #classifier
        output = self.classifier(pooled_output)
        return encoder_output, pooled_output, output





class ViltClassifer(nn.Module):
    def __init__(self, config):
        super(ViltClassifer).__init__()
        self.fc = nn.linear(config.hidden_size, config.hidden_size*2)
        self.norm = nn.LayerNorm(config.hidden_size*2)
        self.activation = config.hidden_act

    def forward(self,x):
        output = self.fc(x)
        output = self.norm(output)
        output = self.activation(output)
        return output





    
class ViltEmbeddings(nn.Module):
    def __init__(self,config):
        #super(ViltEmbeddings).__init__()
        super().__init__()
         # text embeddings
        self.text_embeddings = BERTEmbeddings(config)
        # patch embeddings
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        image_size, patch_size = config.image_size, config.patch_size
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # num_patches = config.num_patches
        self.position_embeddings = nn.Paramete(torch.zeros(1, num_patches + 1, config.hidden_size))
        # modality type embedding
        self.token_type_embeddings = nn.Embedding(config.modality_type_vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        # self.fc = nn.

        def visual_embed(self,pixel_values, pixel_mask, mask_image_length=200):
            """ """
            pass

        def forward(self, 
                        input_ids, attention_mask, token_type_ids,
                         pixel_values, pixel_mask,image_token_type_idx=1):
            # 1. text embeddings
            text_embeds = self.text_embeddings(
                input_ids = input_ids, token_type_ids= token_type_ids )

            # 2. patch embeddings
            """if use clip, change code here
            for example: 
                import clip
                model, preprocess = clip.load("ViT-B/32", device=device)
                image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
                image_embeds = model.encode_image(image)
            """
            image_embeds, image_masks, patch_index = image_embeds(
                pixel_values , pixel_mask, max_image_length=self.config.max_iamge_length )
            
            # 3. add modality type embedding
            text_embeds = text_embeds + self.token_type_embeddings(
                torch.zeros_like(attention_mask,dtype=torch.long, device = text_embeds.device))

            image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx, dtype=torch.long, device=image_embeds.device))

            # 4. concat
            embeddings = torch.cat([text_embeds, image_embeds], dim =1)
            masks = torch.cat([attention_mask, image_masks], dim=1)

            return embeddings, masks


class ViltEncoder(nn.Module):
    def __init__(self,config) :
        super(ViltEncoder).__init__()
        self.config = config
        layer = ViltLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers



class ViltLayer(nn.Module):
    def __init__(self, config):
        super(ViltLayer).__init__()
        self.attention = MultiHeadAttention(config)


class ViltPooler(nn.Module):
    def __init__(self,config):
        super(ViltPooler).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        """taking the hidden state corresponding to the first token."""
        first_token_tensor = hidden_states[:,0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

if __name__ == "__main__":
    config = ViltConfig()
    print(config.vocab_size)
    model = ViltModel(config)
    path1 = 'Dataset/questions/v2_OpenEnded_mscoco_val2014_questions.json'
    path2 = 'Dataset/annotations/v2_mscoco_val2014_annotations.json'
    questions, annotations = read_data(path1,path2,config)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    dataset = VQADataset( questions, annotations,processor,config)
    
    encoder_output, pooled_output, output = model(*dataset)
    print(encoder_output, pooled_output, output)

