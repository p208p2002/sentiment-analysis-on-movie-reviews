import csv
from transformers import AlbertTokenizer

class MR_Data:
    def __init__(self, is_train_data, raw_data):
        self.raw_data = raw_data
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
                    new_row = [PhraseId,SentenceId,Phrase,Sentiment,toBertIds]
                else:
                    PhraseId,SentenceId,Phrase = row
                    idput_ids = toBertIds(Phrase)
                    new_row = [PhraseId,SentenceId,Phrase,toBertIds]
                new_rows.append(new_row)
            return cls(is_train_data,new_rows)

    @property
    def total_topic(self):
        return len(self.raw_data) 
        
if __name__ == "__main__":
    TrainData = MR_Data.load_data('dataset/train.tsv')
    # TestData = MR_Data.load_data('dataset/test.tsv')
    print('TrainData.total_topic:',TrainData.total_topic)
    # print('TestData.total_topic:',TestData.total_topic)