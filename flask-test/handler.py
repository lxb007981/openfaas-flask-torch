import torch
from transformers import BertTokenizer
from model import Bert_Model
from model import NN

# ###################################################################

PATH_TO_STATE_DICT = 'epoch1.pth'
TOKENIZER_PRETRAIN_PATH = 'bert_tokenizer_pretrained/'
BERT_PRETRAIN_PATH = 'bert_pretrained/'

# ###################################################################
'''
mode = 'bert'
device = torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PRETRAIN_PATH)
encoder_model = Bert_Model(device=device, mode=mode, pretrain_path=BERT_PRETRAIN_PATH)
model = NN(encoder_model, output_dimensions=2).to(device)
'''
#model.eval()
#state_dict = torch.load(PATH_TO_STATE_DICT, map_location=device)
#model.load_state_dict(state_dict)

def handle(event, context):
    test_input = event.body
    
    #input = [tokenizer.encode_plus(test_input, max_length=512, add_special_tokens=True, return_token_type_ids=True,
    #                       return_attention_mask=True, padding='max_length', return_tensors='pt').to(device)]
    #decision = torch.argmax(model(input), dim=1).item()
    decision = 1
    return {
        "statusCode": 200,
        "body": test_input,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        }
    }
