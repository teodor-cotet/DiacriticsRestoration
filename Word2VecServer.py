from definitions.Word2VecService import Processor
from thrift.transport import TSocket, TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec 
from utils import split_sentences
from typing import Dict, List
from os.path import dirname

class Word2VecServer:
    
    def __init__(self, port: int):
        self.port = port
        processor = Processor(self)
        transport = TSocket.TServerSocket(port=port)
        tfactory = TTransport.TBufferedTransportFactory()
        pfactory = TBinaryProtocol.TBinaryProtocolFactory()

        self.server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    
    def serve(self):
        self.server.serve()
    
    def trainModel(self, inputFilePath: str, noEpochs:int, layerSize: int):
        print("Starting training: ", inputFilePath)
        try:
            sentences = [sent for sent in split_sentences(inputFilePath)]
        except:
            print("File does not exist!")
            return
        folder = dirname(inputFilePath)
        model = Word2Vec(sentences, size=layerSize, window=5, min_count=5, iter=noEpochs, workers=8)
        path = folder + "/word2vec.model"
        model.wv.save_word2vec_format(path, binary=False)
        print("Model saved to ", path)
       
    def loadModel(self, path: str) -> Dict[str, List]:
        print("Loading model: ", path)
        model = KeyedVectors.load_word2vec_format(path + "/word2vec.model", binary=True)
        return {word: model[word] for word in model.vocab}
        
if __name__ == "__main__":
    server = Word2VecServer(9090)
    print("Starting server..")
    server.serve()
    
    # vocab = server.loadModel("resources/Word2Vec/glove/glove.6B.100d.txt")
    # print(len(vocab))
    