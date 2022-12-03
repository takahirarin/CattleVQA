from __future__ import absolute_import
import torch
import torch.nn as nn
from text_preprocess import BERTEmbeddings
from img_preprocess import PatchEmbeddings
from vilt_config import ViltConfig
import copy
from transformers.modeling_utils import  ModuleUtilsMixin
from transformer_block import MultiHeadAttention, FFN, SubLayerConnection

from transformers import ViltProcessor
from clip_introduce_dataset import VQADataset, read_data, collate_fn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm


from transformers import CLIPVisionModel

from transformers import logging
from transformers import logging



class ViltModel(nn.Module):
    def __init__(self, config: ViltConfig):
        """"config: 'ViltConfig' instance """
        super(ViltModel, self).__init__()
        self.embeddings = ViltEmbeddings(config)
        self.encoder = ViltEncoder(config)
        self.pooler = ViltPooler(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = ViltClassifer(config)


    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            # if config.is_decoder:
            #     batch_size, seq_length = input_shape
            #     seq_ids = torch.arange(seq_length, device=device)
            #     causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            #     # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            #     # causal and attention masks must have same type with pytorch version < 1.3
            #     causal_mask = causal_mask.to(attention_mask.dtype)

            #     if causal_mask.shape[1] < attention_mask.shape[1]:
            #         prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            #         causal_mask = torch.cat(
            #             [
            #                 torch.ones(
            #                     (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
            #                 ),
            #                 causal_mask,
            #             ],
            #             axis=-1,
            #         )

            #     extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            # else:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

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
        # extended_attention_mask = ModuleUtilsMixin.get_extended_attention_mask(attention_mask, input_shape)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(masks, input_shape, device)
        encoder_output = self.encoder(embeddings, extended_attention_mask)
        sequence_output = encoder_output[-1]
        pooled_output = self.pooler(sequence_output)

        #classifier
        output = self.classifier(pooled_output)
        return encoder_output, pooled_output, output





class ViltClassifer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.hidden_size, config.hidden_size*2)
        self.norm = nn.LayerNorm(config.hidden_size*2)
        self.activation = nn.GELU()

    def forward(self,x):
        output = self.fc(x)
        output = self.norm(output)
        output = self.activation(output)
        return output



class ViltEncoder(nn.Module):
    def __init__(self,config) :
        super().__init__()
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
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.transformer_block = FFN(config)
        self.shortcut = SubLayerConnection(config)
    def forward(self,hidden_states,attention_mask):
        attention_output = self.attention(hidden_states, hidden_states, hidden_states,
            config.num_attention_heads, config.hidden_size, attention_mask)
        FFN_output = self.transformer_block(attention_output)
        layer_output = self.shortcut(FFN_output, attention_output)
        return layer_output

class ViltPooler(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        """taking the hidden state corresponding to the first token."""
        first_token_tensor = hidden_states[:,0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ViltEmbeddings(nn.Module):
    def __init__(self,config):
        logging.set_verbosity_error()
        logging.set_verbosity_warning()
        #super(ViltEmbeddings).__init__()
        super().__init__()
         # text embeddings
        self.text_embeddings = BERTEmbeddings(config)
        # patch embeddings
        self.patch_embeddings = PatchEmbeddings(config)
        self.vision_embeddings = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        num_patches = self.patch_embeddings.num_patches
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        # modality type embedding
        self.token_type_embeddings = nn.Embedding(config.modality_type_vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def mask_embed(self,pixel_values, pixel_mask, max_image_length=200):
        _, _, ph, pw = self.patch_embeddings.projection.weight.shape

        x = self.patch_embeddings(pixel_values)
        x_mask = pixel_mask[:, None, :, :].float()
        x_mask = nn.functional.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]

        batch_size, num_channels, height, width = x.shape
        patch_dim = self.config.image_size // self.config.patch_size
        spatial_pos = self.position_embeddings[:, 1:, :].transpose(1, 2).view(1, num_channels, patch_dim, patch_dim)
        pos_embed = torch.cat(
            [
                nn.functional.pad(
                    nn.functional.interpolate(
                        spatial_pos,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=True,
                    ),
                    (0, width - w, 0, height - h),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )

        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
         # Set `device` here, otherwise `patch_index` will always be on `CPU` and will fail near the end for torch>=1.13
        patch_index = torch.stack(
            torch.meshgrid(torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1]), indexing="ij"), dim=-1
        ).to(device=x_mask.device)
        patch_index = patch_index[None, None, :, :, :]
        patch_index = patch_index.expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
        patch_index = patch_index.flatten(1, 3)
        x_mask = x_mask.flatten(1)

        if max_image_length < 0 or max_image_length is None or not isinstance(max_image_length, int):
            # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrinked)
            # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
            # if self.patch_size = 32, 25 * 41 = 1025
            # if res is 384 x 640, 12 * 20 = 240
            effective_resolution = x_h * x_w
            max_image_length = effective_resolution.max()
        else:
            effective_resolution = x_h * x_w
            max_image_length = min(effective_resolution.max(), max_image_length)

        valid_idx = x_mask.nonzero(as_tuple=False)
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
        non_valid_row_idx = [non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows]
        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_length - v for v in valid_nums]

        select = list()
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
            if p <= 0:
                valid_choice = torch.multinomial(torch.ones(v).float(), max_image_length)
                select.append(valid_row_idx[i][valid_choice])
            else:
                pad_choice = torch.multinomial(torch.ones(nv).float(), p, replacement=True)
                select.append(torch.cat([valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0))

        select = torch.cat(select, dim=0)
        x = x[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)
        x_mask = x_mask[select[:, 0], select[:, 1]].view(batch_size, -1)
        # `patch_index` should be on the same device as `select` (for torch>=1.13), which is ensured at definition time.
        patch_index = patch_index[select[:, 0], select[:, 1]].view(batch_size, -1, 2)
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = torch.cat(
            (self.position_embeddings[:, 0, :][:, None, :].expand(batch_size, -1, -1), pos_embed), dim=1
        )
        x = x + pos_embed
        x = self.dropout(x)

        x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)

        #return x, x_mask, (patch_index, (height, width))
        return x_mask
        




    def forward(self,input_ids, attention_mask, token_type_ids,
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

                モジュール自体を少し変更しました
            """
            not_use, image_embeds,  = self.vision_embeddings(pixel_values)

            image_masks = self.mask_embed(pixel_values , pixel_mask, max_image_length=self.config.max_image_length)
            
            # 3. add modality type embedding
            # text_embeds = text_embeds + self.token_type_embeddings(
            #     torch.zeros_like(attention_mask,dtype=torch.long, device = text_embeds.device))

            image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx, dtype=torch.long, device=image_embeds.device))

            # 4. concat
            embeddings = torch.cat([text_embeds, image_embeds], dim =1)
            masks = torch.cat([attention_mask, image_masks], dim=1)
            return embeddings, masks

if __name__ == "__main__":
    config = ViltConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViltModel(config).to(device)
    path1 = 'Dataset/questions/v2_OpenEnded_mscoco_val2014_questions.json'
    path2 = 'Dataset/annotations/v2_mscoco_val2014_annotations.json'
    questions, annotations = read_data(path1,path2,config)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    dataset = VQADataset( questions[:32], annotations[:32],config)
    
    dataloader = DataLoader(dataset, collate_fn=collate_fn,batch_size =config.batch_size,shuffle=True )
    for datas in tqdm(dataloader):
        data = {k:v.to(device) for k,v in datas.items()}
        data.pop('labels')
        print(data.keys())
        encoder_output, pooled_output, output = model(**data)
        print(encoder_output, pooled_output, output)

