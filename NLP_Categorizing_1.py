
# coding: utf-8

# # WOMEN WHO CODE - DATA ANALYTICS WITH PYTHON (INTERMEDIATE)
# 
# ## Introduction to Natural Language Processing
# ## Categorizing, Tagging and Classifying
# 
# Based on Natural Language Processing with Python (Authors: Steven Bird, Ewan Klein & Edward Loper)
# Published by O'Reilly

# In[1]:

import nltk


# nltk.download()
# 
# Download the NLTK Book Collection (30 compressed files about 100Mb disk space) to complete the exercises

# In[2]:

from nltk.book import *


# ### COLLOCATIONS AND N-GRAMS
# 
# A collocation is a sequence of words that occur together unsually often. Thus *red wine* is a collocation.
# A characteristic of collocations is that they are resistant to substitution with words that have similar senses; for example *maroon wine* sounds very odd.

# In[3]:

from nltk.util import bigrams


# In[4]:

from nltk.collocations import *


# In[5]:

text4.collocations()


# In[6]:

bigram_measures = nltk.collocations.BigramAssocMeasures()


# In[7]:

trigram_measures = nltk.collocations.TrigramAssocMeasures()


# In[8]:

finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))


# For example, the top ten bigram collocations in Genesis are listed below

# In[9]:

finder.nbest(bigram_measures.pmi, 10) 


# In[10]:

from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# In[11]:

words = [w.lower() for w in webtext.words('grail.txt')]


# In[12]:

bcf = BigramCollocationFinder.from_words(words)


# In[13]:

bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4)


# Eliminating Stopwords

# In[14]:

from nltk.corpus import stopwords


# In[15]:

stopset = set(stopwords.words('english'))


# In[16]:

filter_stops = lambda w: len(w) < 3 or w in stopset


# In[17]:

bcf.apply_word_filter(filter_stops)


# In[18]:

bcf.nbest(BigramAssocMeasures.likelihood_ratio, 4)


# ### TAGGING
# 
# A part-of-speech tagger, or POS tagger, processes a sequence of words, and attaches
# a part of speech tag to each word.

# In[19]:

text = nltk.word_tokenize("And now for something completely different")


# In[20]:

nltk.pos_tag(text)


# NLTK provides documentation for each tag, which can be queried using the tag

# In[21]:

nltk.help.upenn_tagset('RB')


# Let's see how NLTK behaves when working with homonyms which are words that have the same spelling but different meanings and origins. NLTK will analyze the whole regular expression.

# In[22]:

text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")


# In[23]:

nltk.pos_tag(text)


# ### Exercise:
# 
# Create an example of a regular expression using homonyms.

# ### TAGGING CORPORA
# 
# A tagged token is represented using a tuple consisting of the token and the tag. Use the function str2tuple():

# In[24]:

tagged_token = nltk.tag.str2tuple('fly/NN')


# In[25]:

tagged_token


# In[26]:

tagged_token[0]


# In[27]:

tagged_token[1]


# You can create a complex tagged sentence...

# In[28]:

sent = '''
The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
other/AP topics/NNS ,/, AMONG/IN them/PPO the/AT Atlanta/NP and/CC
Fulton/NP-tl County/NN-tl purchasing/VBG departments/NNS which/WDT it/PPS
said/VBD ``/`` ARE/BER well/QL operated/VBN and/CC follow/VB generally/RB
accepted/VBN practices/NNS which/WDT inure/VB to/IN the/AT best/JJT
interest/NN of/IN both/ABX governments/NNS ''/'' ./.
'''


# In[29]:

[nltk.tag.str2tuple(t) for t in sent.split()]


# In[30]:

nltk.corpus.brown.tagged_words()


# Several of the corpora included with NLTK have been tagged for their part-of-speech.Other corpora use a variety of formats for storing part-of-speech tags. NLTK's corpus readers provide a uniform interface so that you don't have to be concerned with the different file formats.

# ### USING A UNIVERSAL TAGSET
# 
# Tag   Meaning	            Examples   
# ADJ	  adjective:	        new, good, high, special, big, local   
# ADV	  adverb:	            really, already, still, early, now   
# CNJ	  conjunction:        and, or, but, if, while, although  
# DET	  determiner:	        the, a, some, most, every, no  
# EX	  existential:	        there, there's  
# FW	  foreign word:        dolce, ersatz, esprit, quo, maitre  
# MOD	  modal verb:	        will, can, would, may, must, should  
# N	  noun:	            year, home, costs, time, education  
# NP	  proper noun:	        Alison, Africa, April, Washington  
# NUM	  number:	            twenty-four, fourth, 1991, 14:24  
# PRO	  pronoun:	            he, their, her, its, my, I, us       
# P	  preposition:	        on, of, at, with, by, into, under     
# TO	  the word to:	        to   
# UH	  interjection:        ah, bang, ha, whee, hmpf, oops   
# V	  verb:	            is, has, get, do, make, see, run    
# VD	  past tense:	        said, took, told, made, asked    
# VG	  present participle:	making, going, playing, working   
# VN	  past participle:	    given, taken, begun, sung      
# WH	  wh determiner:	    who, which, when, what, where, how    
# 

# In[31]:

from nltk.corpus import brown


# the latest version of simplified tag is to map them to the universal tagset (https://code.google.com/p/universal-pos-tags/).

# In[32]:

brown.tagged_words(tagset='universal')[:10]


# In[33]:

brown_news_tagged = brown.tagged_words(categories = 'news', tagset='universal')


# In[34]:

tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)


# In[35]:

tag_fd.keys()


# Plot the above frequency distribution using tag_fd.plot(cumulative=True). What percentage of words are tagged using the first five tags of the above list?

# In[ ]:

# tag_fd.plot(cumulative=True)


# ### TRAINING A UNI-GRAM Tagging
# 
# - Unigram taggers are based on a simple statistical algorithm: for each token, assign the tag that is most likely for that particular token. 
# - For example, it will assign the tag JJ to any occurrence of the word frequent, since frequent is used as an adjective (e.g., a frequent word) more often than it is used as a verb (e.g., I frequent this cafe).
# - A unigram tagger behaves just like a lookup tagger
# - In the following code sample, we train a unigram tagger, use it to tag a sentence, and then evaluate

# In[43]:

from nltk.corpus import brown


# In[44]:

brown_tagged_sents = brown.tagged_sents(categories='news')


# In[45]:

brown_sents = brown.sents(categories='news')


# In[47]:

unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)


# In[48]:

unigram_tagger.tag(brown_sents[2007])


# In[49]:

unigram_tagger.evaluate(brown_tagged_sents)


# We train a UnigramTagger by specifying tagged sentence data as a parameter when we
# initialize the tagger. The training process involves inspecting the tag of each word and
# storing the most likely tag for any word in a dictionary that is stored inside the tagger.

# ### CREATING A TRAINING AND A TESTING DATA SETS
# 
# A tagger that simply memorized its training data and made no attempt to construct a general model would get a perfect
# score, but would be useless for tagging new text. Instead, we should split the data,
# training on 90% and testing on the remaining 10%:

# In[50]:

size = int(len(brown_tagged_sents) * 0.9)


# In[51]:

size


# In[52]:

train_sents = brown_tagged_sents[:size]


# In[53]:

test_sents = brown_tagged_sents[size:]


# In[54]:

unigram_tagger = nltk.UnigramTagger(train_sents)


# In[55]:

unigram_tagger.evaluate(test_sents)


# ### GENERAL N-GRAM TAGGING
# 
# An n-gram tagger is a generalization of a unigram tagger whose context is the current
# word together with the part-of-speech tags of the n-1 preceding tokens.
# The **NgramTagger class** uses a tagged training corpus to determine which part-of-speech
# tag is most likely for each context. Here we see a special case of an n-gram tagger,
# namely a bigram tagger

# In[56]:

bigram_tagger = nltk.BigramTagger(train_sents)


# In[57]:

bigram_tagger.tag(brown_sents[2007])


# In[58]:

unseen_sent = brown_sents[4203]


# In[59]:

bigram_tagger.tag(unseen_sent)


# Notice that the bigram tagger manages to tag every word in a sentence it saw during
# training, but does badly on an unseen sentence. As soon as it encounters a new word, it is unable to assign a tag. It cannot tag the following word, even if it was seen during training, simply because it never saw it during training with a None tag on the previous word. Consequently, the tagger fails to tag the rest of the
# sentence. Its overall accuracy score is very low.

# In[60]:

bigram_tagger.evaluate(test_sents)


# ### Conclusion:
# 
# As n gets larger, the specificity of the contexts increases, as does the chance that the
# data we wish to tag contains contexts that were not present in the training data. This
# is known as the sparse data problem, and is quite pervasive in NLP. As a consequence,
# there is a trade-off between the accuracy and the coverage of our results (and this is
# related to the precision/recall trade-off in information retrieval).

# ### COMBINING TAGGERS
# 
# One way to address the trade-off between accuracy and coverage is to use the more
# accurate algorithms when we can, but to fall back on algorithms with wider coverage
# when necessary. For example, we could combine the results of a bigram tagger, a
# unigram tagger, and a default tagger, as follows:
# 1. Try tagging the token with the bigram tagger.
# 2. If the bigram tagger is unable to find a tag for the token, try the unigram tagger.
# 3. If the unigram tagger is also unable to find a tag, use a default tagger.
# Most NLTK taggers permit a backoff tagger to be specified. The backoff tagger may
# itself have a backoff tagger. These taggers inherit from SequentialBackoffTagger, which allows them to be chained together for greater accuracy. 
# 
# more info at: http://streamhacker.com/2008/11/03/part-of-speech-tagging-with-nltk-part-1/

# In[61]:

t0 = nltk.DefaultTagger('NN')


# In[62]:

t1 = nltk.UnigramTagger(train_sents, backoff=t0)


# In[63]:

t2 = nltk.BigramTagger(train_sents, backoff=t1)


# In[64]:

t2.evaluate(test_sents)


# A useful method to tag unknown words based on context is to limit the vocabulary of
# a tagger to the most frequent n words, and to replace every other word with a special
# word UNK.
# During training, a unigram tagger will probably learn that UNK is usually a noun. However, the n-gram taggers will detect contexts in which it has some other tag. For example, if the preceding word is to (tagged TO), then UNK will probably be tagged as a verb.

# ### STORE AND RETRIEVE A MODEL
# 
# Training a tagger on a large corpus may take a significant time. Instead of training a
# tagger every time we need one, it is convenient to save a trained tagger in a file for later
# reuse. Let’s save our tagger t2 to a file t2.pkl

# >>> from cPickle import dump
# >>> output = open('t2.pkl', 'wb')
# >>> dump(t2, output, -1)
# >>> output.close()

# Now, in a separate Python process, we can load our saved tagger:
# >>> from cPickle import load
# >>> input = open('t2.pkl', 'rb')
# >>> tagger = load(input)
# >>> input.close()

# Now let’s check that it can be used for tagging:
# >>> text = """The board's action shows what free enterprise
# ... is up against in our complex maze of regulatory laws ."""
# >>> tokens = text.split()
# >>> tagger.tag(tokens)
