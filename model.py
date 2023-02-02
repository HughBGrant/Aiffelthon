import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatBot:
    def __init__(self, model_name='skt/kogpt2-base-v2', model_path='model.pth'):
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.usr_token = '<usr>'
        self.pad_token = '<pad>'
        self.sys_token = '<sys>'
        self.unk_token = '<unk>'
        self.mask_token = '<mask>'
        self.max_length = 256
        self.max_turns = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "skt/kogpt2-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                    bos_token=self.bos_token, eos_token=self.eos_token, unk_token=self.unk_token,
                    pad_token=self.pad_token, mask_token=self.mask_token, model_max_length=self.max_length)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.load_state_dict(torch.load(model_path), map_location=torch.device('cpu'))
        self.history_limit = [self.bos_token]
        self.chat_history = []
        
    def get_reply(self, user_message):
        # save message from the user
        self.chat_history.append({
            'text': user_message, 
          '  time': str(datetime.datetime.now().time().replace(microsecond=0))
        })
        
        if len(self.history_limit) == self.max_turns * 2 + 1:
            self.history_limit = self.history_limit[: 1] + self.history_limit[3: ]
            
        user_message_pull = self.usr_token + user_message + self.sys_token
        
        self.history_limit.append(user_message_pull)
        
        # encode the new user message to be used by our model
        message_ids = self.tokenizer.encode(''.join(self.history_limit), return_tensors="pt").to(self.device)

        with torch.no_grad():
            reply_ids = self.model.generate(
                        message_ids,
                        max_length=256,
                        top_k=100,
                        top_p=0.95,
                        do_samples=True,
                        temperature=0.8,
                        max_new_tokens=30,
                        eos_token_id=self.tokenizer.eos_token_id
            )
            
        decoded_ids = reply_ids[0, message_ids.shape[-1]: ]
        if decoded_ids[-1] == self.tokenizer.eos_token_id:
            decoded_ids = decoded_ids[: -1]
            
        decoded_message = self.tokenizer.decode(decoded_ids)
        
        self.history_limit.append(decoded_message)
        
        # save reply from the bot
        self.chat_history.append({
            'text':decoded_message, 
            'time':str(datetime.datetime.now().time().replace(microsecond=0))
        })

        return decoded_message