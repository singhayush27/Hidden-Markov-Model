import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt # show graph
from sklearn.model_selection import GroupShuffleSplit
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = pd.read_csv("final2.csv")
data = data.fillna(method="ffill")
data = data.rename(columns={'Sentence #': 'sentence'})
data.head(5)

tags = list(set(data.POS.values)) #Read POS values
words = list(set(data.Word.values))
len(tags), len(words)


y = data.POS
X = data.drop('POS', axis=1)

gs = GroupShuffleSplit(n_splits=2, test_size=.33, random_state=42)
train_ix, test_ix = next(gs.split(X, y, groups=data['sentence']))

data_train = data.loc[train_ix]
data_test = data.loc[test_ix]

data_train

data_test

tags = list(set(data_train.POS.values)) #Read POS values
words = list(set(data_train.Word.values))
len(tags), len(words)


dfupdate = data_train.sample(frac=.2, replace=False)
dfupdate.Word = 'UNKNOWN'
data_train.update(dfupdate)
words = list(set(data_train.Word.values))
# Convert words and tags into numbers
word2id = {w: i for i, w in enumerate(words)}
tag2id = {t: i for i, t in enumerate(tags)}
id2tag = {i: t for i, t in enumerate(tags)}
len(tags), len(words)



count_tags = dict(data_train.POS.value_counts())
count_tags_to_words = data_train.groupby(['POS']).apply(lambda grp: grp.groupby('Word')['POS'].count().to_dict()).to_dict()
count_init_tags = dict(data_train.groupby('sentence').first().POS.value_counts())

# TODO use panda solution
count_tags_to_next_tags = np.zeros((len(tags), len(tags)), dtype=int)
sentences = list(data_train.sentence)
pos = list(data_train.POS)
for i in range(len(sentences)) :
    if (i > 0) and (sentences[i] == sentences[i - 1]):
        prevtagid = tag2id[pos[i - 1]]
        nexttagid = tag2id[pos[i]]
        count_tags_to_next_tags[prevtagid][nexttagid] += 1



mystartprob = np.zeros((len(tags),))
mytransmat = np.zeros((len(tags), len(tags)))
myemissionprob = np.zeros((len(tags), len(words)))
num_sentences = sum(count_init_tags.values())
sum_tags_to_next_tags = np.sum(count_tags_to_next_tags, axis=1)
for tag, tagid in tag2id.items():
    floatCountTag = float(count_tags.get(tag, 0))
    mystartprob[tagid] = count_init_tags.get(tag, 0) / num_sentences
    for word, wordid in word2id.items():
        myemissionprob[tagid][wordid]= count_tags_to_words.get(tag, {}).get(word, 0) / floatCountTag
    for tag2, tagid2 in tag2id.items():
        mytransmat[tagid][tagid2]= count_tags_to_next_tags[tagid][tagid2] / sum_tags_to_next_tags[tagid]




model = hmm.CategoricalHMM(n_components=len(tags), algorithm='viterbi', random_state=42)
model.startprob_ = mystartprob
model.transmat_ = mytransmat
model.emissionprob_ = myemissionprob



# data_test=pd.read_csv('test.csv')
data_test.loc[~data_test['Word'].isin(words), 'Word'] = 'UNKNOWN'
word_test = list(data_test.Word)
samples = []
for i, val in enumerate(word_test):
    samples.append([word2id[val]])
    
# TODO use panda solution
lengths = []
count = 0
sentences = list(data_test.sentence)
for i in range(len(sentences)) :
    if (i > 0) and (sentences[i] == sentences[i - 1]):
        count += 1
    elif i > 0:
        lengths.append(count)
        count = 1
    else:
        count = 1


# This code is very slow
pos_predict = model.predict(samples, lengths)
pos_predict


tags_test = list(data_test.POS)
pos_test = np.zeros((len(tags_test), ), dtype=int)
for i, val in enumerate(tags_test):
    pos_test[i] = tag2id[val]
len(pos_predict), len(pos_test), len(samples), len(word_test)



def reportTest(y_pred, y_test):
    print("The accuracy is {}".format(accuracy_score(y_test, y_pred))) 
    print("The precision is {}".format(precision_score(y_test, y_pred, average='weighted'))) 
    print("The recall is {}".format(recall_score(y_test, y_pred, average='weighted'))) 
    print("The F1-Score is {}".format(f1_score(y_test, y_pred, average='weighted')))

min_length = min(len(pos_predict), len(pos_test))

reportTest(pos_predict[:min_length], pos_test[:min_length])


with open(r"test.txt") as f:
    testlines=f.readlines()
a=0
testlis=[]
dictionary=dict()
for i in testlines:
    splitted=i.strip('\n').split('\t')
    sentance=f'sentence : {a}'
    if(splitted[0]!=''):
        testlis.append([sentance,*splitted])
    else:
        a+=1
dftest=pd.DataFrame(testlis)
dftest=dftest.rename(columns={0:'sentence'})
dftest=dftest.rename(columns={1:'Word'})



data_test=dftest
data_test.loc[~data_test['Word'].isin(words), 'Word'] = 'UNKNOWN'
word_test = list(data_test.Word)
samples = []
for i, val in enumerate(word_test):
    samples.append([word2id[val]])
    
# TODO use panda solution
lengths = []
count = 0
sentences = list(data_test.sentence)
for i in range(len(sentences)) :
    if (i > 0) and (sentences[i] == sentences[i - 1]):
        count += 1
    elif i > 0:
        lengths.append(count)
        count = 1
    else:
        count = 1
# This code is very slow
pos_predict = model.predict(samples, lengths)
pos_predict


dic={}
for i,j in tag2id.items():
    dic[j]=i


samples
anotherdic={}
results=[]
for i , j in word2id.items():
    anotherdic[j]=i
li=[]
for s in samples:
    li.append(anotherdic[s[0]])
for i in pos_predict:
    results.append(dic[i])



ResultDataFrame=pd.DataFrame()
ResultDataFrame['Words']=li[:-25]
ResultDataFrame['Tags']=results
ResultDataFrame.to_csv('ResultTags.csv')
print(ResultDataFrame.to_string())