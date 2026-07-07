from keras.models import load_model
from keras.layers import Dense
import numpy as np

_original_dense_from_config = Dense.from_config


def _dense_from_config(cls, config):
    config = dict(config)
    config.pop('quantization_config', None)
    return _original_dense_from_config(config)


Dense.from_config = classmethod(_dense_from_config)

model=load_model('model.h5')

output_dict={0:['Negative 😢 ','static/img/negative.png'],1:['Neutral 😐','static/img/neutral.png'],2:['Positive 😊 ','static/img/positive.png']}

# f=open('D:/Skills/Emoji Classifier/glove.6B.50d.txt',encoding='utf-8')
f = open("glove.6B/glove.6B.50d.txt", encoding="utf-8")
embedding_index={}

for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float')
    embedding_index[word]=coefs

f.close()

def embedding_output(X):
    maxLen=50 #max words in sentence
    embedding_out=np.zeros((X.shape[0],maxLen,50))
    for ix in range(X.shape[0]):
        # print(ix)
        try:
            X[ix]=X[ix].split()
        except:
            pass
        
        for ij in range(maxLen):
            # print(ix,ij)
            # print(X[ij:ix])
            # go to every word in the current (ix) sentence
            try:
                embedding_out[ix][ij]=embedding_index[X[ix][ij].lower()]
                # print(X[ix][ij])
            except:
                embedding_out[ix][ij]=np.zeros((50,))
    # print(embedding_out)
    return embedding_out

# model prediction 
# def predict(X):
#     X=np.asarray([X])
#     print(X.shape)
#     X=embedding_output(X)
#     print(model.predict(X))
#     return output_dict[np.argmax(model.predict(X))]

def predict(text):
    # Split the input string into a list of words
    words = text.split()
    
    # Create an empty embedding matrix for 1 sample, maxLen=50, embedding_dim=50
    embedding_out = np.zeros((1, 50, 50)) 
    
    # Map words to their GloVe embeddings
    for ij in range(min(50, len(words))):
        try:
            embedding_out[0][ij] = embedding_index[words[ij].lower()]
        except:
            embedding_out[0][ij] = np.zeros((50,))
            
    # Get the model prediction
    pred_probs = model.predict(embedding_out)
    
    # YAHAN CHANGE KAREIN: Return the dictionary value instead of just the number
    return output_dict[np.argmax(pred_probs)]

if __name__=='__main__':
    print(predict('Lately, doesnt sync between web app and Android reminders well. Still a nice app, but that is an inconvenience.'))
