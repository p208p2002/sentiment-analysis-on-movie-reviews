import requests

# 下載必要的模型檔案
# Download needed model files

urls = ['https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-pytorch_model.bin','https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-spiece.model']
filenames = ['albert-large-pytorch_model.bin','albert-large-spiece.model']

for i,url in enumerate(urls):
    r = requests.get(url)
    with open('./'+filenames[i], 'wb') as f:
        f.write(r.content)
