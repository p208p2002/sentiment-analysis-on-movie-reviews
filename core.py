import csv
from transformers import AlbertTokenizer
import torch
from torch.utils.data import TensorDataset

def makeTorchDataSet(mr_data_class,is_train_data = True):
    all_input_ids = []
    all_answer_lables = []
    max_input_length = 0
    for d in mr_data_class.data:
        input_ids = d[-1]
        if(len(input_ids)>max_input_length):
            max_input_length = len(input_ids)
        all_input_ids.append(input_ids)
        if(is_train_data):
            Sentiment = d[-2]
            all_answer_lables.append(int(Sentiment))
    
    for input_ids in all_input_ids:
        while(len(input_ids)<max_input_length):
            input_ids.append(0)
        assert len(input_ids) == max_input_length

    if(is_train_data):
        torch_input_ids = torch.tensor([ids for ids in all_input_ids], dtype=torch.long)
        torch_answer_lables = torch.tensor([answer_lable for answer_lable in all_answer_lables], dtype=torch.long)
        return TensorDataset(torch_input_ids,torch_answer_lables)

    return TensorDataset(torch_input_ids)

class MR_Data:
    def __init__(self, is_train_data, data):
        self.data = data
        self.is_train_data = is_train_data
    
    @classmethod
    def load_data(cls, path, is_train_data = True):
        tokenizer = AlbertTokenizer.from_pretrained('albert-large-spiece.model')
        def toBertIds(q_input):
            return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q_input)))
        with open(path) as f:
            rows = csv.reader(f, delimiter='\t')
            rows = list(rows)            
            rows.pop(0)
            new_rows = []
            for row in rows:
                new_row = []
                if(is_train_data):
                    PhraseId,SentenceId,Phrase,Sentiment = row
                    idput_ids = toBertIds(Phrase)
                    new_row = [PhraseId,SentenceId,Phrase,Sentiment,idput_ids]
                else:
                    PhraseId,SentenceId,Phrase = row
                    idput_ids = toBertIds(Phrase)
                    new_row = [PhraseId,SentenceId,Phrase,idput_ids]
                new_rows.append(new_row)
            return cls(is_train_data,new_rows)
    
    @property
    def total_topic(self):
        return len(self.data) 
        
if __name__ == "__main__":
    TrainData = MR_Data.load_data('dataset/train.tsv')    
    print('TrainData.total_topic:',TrainData.total_topic)
    # TestData = MR_Data.load_data('dataset/test.tsv')
    # print('TestData.total_topic:',TestData.total_topic)