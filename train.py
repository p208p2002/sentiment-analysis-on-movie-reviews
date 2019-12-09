from core import MR_Data, makeTorchDataSet, makeTorchDataLoader
if __name__ == "__main__":
    TrainData = MR_Data.load_data('dataset/train.tsv')
    TrainDataset = makeTorchDataSet(TrainData)
    TrainDataLoader = makeTorchDataLoader(TrainDataset)