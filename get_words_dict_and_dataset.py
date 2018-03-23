import numpy as np
import nltk
import re
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json

def clean_and_get_dict(docs):
	num_docs = len(docs)
	clean_doc_words = []
	idx = 0
	word_to_index = {}
	l = WordNetLemmatizer()
	corpus = open("data/english_words_corpus.txt", "w+")
	val = open("data/english_words_val.txt", "w+")
	test = open("data/english_words_test.txt", "w+")

	for i,doc in enumerate(docs):
		#remove non-words and stopwords, change all words to lowercase, and lemmatize for both verbs and nouns
		
		clean_doc = re.sub("[^a-zA-Z]+", " ", doc).lower()
		#print(clean_doc)
		doc_words=[]
		if i in range(1000):
			corpus.write(clean_doc + '\n')
		elif i in range(1000,1250):
			val.write(clean_doc+ '\n')
		elif i in range(1250,1500):
			test.write(clean_doc+ '\n')


		for w in clean_doc.split(' '):
			#build list of clean, lemmatized words
			if w!='':
				doc_words.append(w.lower())
				#build word_to_index dictionary of words
				if w not in word_to_index.keys():
					word_to_index[w] = idx
					idx=idx+1
		#clean_doc = [l.lemmatize(l.lemmatize(s.lower(),pos='n'),pos='v') for s in clean_doc.split(' ') if s!='' and s not in remove_words]
		clean_doc_words.append(doc_words)
	corpus.close()
	val.close()
	test.close()

		#clean_docs.append(" ".join(doc_words))
	return word_to_index

doc_files = np.array(reuters.fileids())
test_mask = ['test' in doc for doc in doc_files]
train_mask = np.array(test_mask,dtype=int) == 0

test_doc_files = doc_files[test_mask]

num_docs = len(doc_files)
num_train = np.sum(train_mask)
num_test = np.sum(test_mask)

all_docs = [" ".join(reuters.words(doc_files[i])) for i in range(num_docs)]
train_docs = np.array(all_docs)[train_mask]

#get stop words to remove from NLTK
remove_words = set(stopwords.words('english'))
print('building dict')
word_to_index = clean_and_get_dict(all_docs[:1500])
print(len(word_to_index.keys()))
with open('data/reuters_vocab.txt', 'w+') as f:
     f.write(json.dumps(word_to_index))
     f.close()

	#print(file_content.lower())
	#print(file_content.lower())
	#corpus.write(file_content.lower())

