import gzip
import shutil
import os
data_root = '/Users/antoniotantorres/workspace/riminder/hansard.36/Release-2001.1a/sentence-pairs/senate/debates/development/training'
files = []
for fs in os.listdir(data_root):
	if fs[-4:] == 'f.gz':
		fl = os.path.join(data_root,fs)
		files.append(fl)
#print(files,len(files))
corpus = open("french_corpus.txt", "wb+")
for i in range(40):
	fil = files[i]
	f = gzip.open(fil, 'rb')
	file_content = f.read()
	#print(file_content.lower())
	fc = file_content.lower()
	g = re.sub("[^a-zA-Z]+", " ", fc)
	print(g)
	corpus.write(g)
	#print(file_content.lower())
	#print(file_content.lower())
	#corpus.write(file_content.lower())
	f.close()
corpus.close()

val = open("french_corpus_val.txt","wb+")
for i in range(45,48):
	fil = files[i]
	f = gzip.open(fil, 'rb')
	file_content = f.read()
	#print(file_content.lower())
	#print(file_content.lower())
	val.write(file_content.lower())
	f.close()
val.close()

test = open("french_corpus_test.txt","wb+")
for i in range(50,53):
	fil = files[i]
	f = gzip.open(fil, 'rb')
	file_content = f.read()
	#print(file_content.lower())
	#print(file_content.lower())
	test.write(file_content.lower())
	f.close()
val.close()