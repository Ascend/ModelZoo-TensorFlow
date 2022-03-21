import pickle
import numpy as np

with open('/home/TestUser07/stackgan/Data/birds/test/char-CNN-RNN-embeddings.pickle', 'rb') as f:  # [2933,10,1024]
    embeddings = pickle.load(f, encoding="latin-1")
    embeddings = np.array(embeddings)   #(2933, 10, 1024)
    embeddings=np.squeeze(embeddings[:,0,:])
    batch_size=64
    length=embeddings.shape[0]
    start=0
    count=0
    while start<length:
        if (start + batch_size) > length:
            end =length
            start=end-batch_size
        else:
            end = start + batch_size
        sampled_embeddings=embeddings[start:end]
        sampled_embeddings.tofile('/home/TestUser07/stackgan/offline_inference/source_binfile/file'+'%02d'%count+'.bin')
        print("doing "+str(count))
        start+=batch_size
        count+=1



