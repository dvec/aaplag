import json
import logging
import os
import numpy as np
import tensorflow as tf
import regex as re
import random
import string

import nn.gpt2.model as model
import nn.gpt2.sample as sample
import nn.gpt2.encoder as encoder

from gensim.summarization import keywords
from gensim.test.utils import datapath, common_texts, get_tmpfile
from gensim.models import Word2Vec, keyedvectors


from deeppavlov import configs, build_model

print('Embeddings are Loading.')
embeddings = keyedvectors.KeyedVectors.load_word2vec_format("wikipedia.6B.100d.model", binary=True)
print('Embeddings are ready.')


class ReplaceableWordsDetector:
    ner_model = None
    stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
                 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',
                 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don',
                 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while',
                 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because',
                 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has',
                 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

    def __init__(self):
        # NER model initner_rus
        if os.path.exists('/root/.deeppavlov'):
            print('NER model is already existing, no need to download it')
            self.ner_model = build_model(configs.ner.ner_ontonotes, download=False)
        else:
            print('Failed to find a NER model. Downloading...')
            self.ner_model = build_model(configs.ner.ner_ontonotes, download=True)

    def get_stopwords_keywords_indicies(self, keywords, text):
        stopwords_keywords_indicies = []
        lowered_text = [word.lower() for word in text]
        for ind, word in enumerate(lowered_text):
            if word in self.stopwords or word in keywords:
                stopwords_keywords_indicies.append(ind)
        return stopwords_keywords_indicies

    def get_replaceable_words(self, text):
        '''
        Get replaceable words from text!

        Parameters:
        text (string): Input text

        Returns:
        replaceable_words_indicies (list): List of replaceable words indicies
        splitted_words (list): List of words

        '''
        result = self.ner_model([text])
        indicies_ner = [ind for ind, val in enumerate(result[1][0]) if val != 'O']
        try:
            indicies_keywords = self.get_keywords_indicies(keywords(text, split=True), result[0][0])
        except Exception as e:
            print(e)
            return [indicies_ner, result[0][0]]

        replaceable_words_indicies = list(set().union(indicies_ner, indicies_keywords))
        return [replaceable_words_indicies, result[0][0]]


def isFitForReplacement(word, word_index, keyword_indicies):
    min_word_len = 3
    if len(word) >= min_word_len and re.match("^[A-Za-z-]*$", word):
        if word_index not in keyword_indicies:
            return True
    return False


def join_words(words, end):
    return ' '.join(words[0:end])


class interact_model:
    ratio = 0.2
    model_name = '117M'
    seed = None
    nsamples = 10
    batch_size = None
    length = 1
    temperature = 1
    top_k = 0
    sess = tf.Session(graph=tf.Graph())

    def __init__(self):
        # MAGIC STUFF BEGIN!

        if self.batch_size is None:
            self.batch_size = 1
        assert self.nsamples % self.batch_size == 0
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        enc = encoder.get_encoder(self.model_name)
        hparams = model.default_hparams()
        with open(os.path.join('models', self.model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if self.length is None:
            self.length = hparams.n_ctx // 2
        elif self.length > hparams.n_ctx:
            raise ValueError(f"can't get samples longer than window size: {hparams.n_ctx}")




        context = tf.placeholder(tf.int32, [self.batch_size, None])
        output = sample.sample_sequence(
            hparams=hparams, length=self.length,
            context=context,
            batch_size=self.batch_size,
            temperature=self.temperature, top_k=self.top_k
        )[:, 1:]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', self.model_name))
        saver.restore(self.sess, ckpt)

            # MAGIC STUFF OVER

        def generate(words, keyword_indicies):
            old_new_words = []
            words = words.copy()
            loss = 0
            for ind in range(len(words)):
                if ind < 10:
                    continue
                elif not isFitForReplacement(words[ind], ind, keyword_indicies) or random.random() > self.ratio:
                    continue
                else:
                    raw_text = join_words(words, ind)
                    generated_words = []
                    context_tokens = enc.encode(raw_text)
                    generated = 0
                    for _ in range(self.nsamples // self.batch_size):
                        out = self.sess.run(output, feed_dict={
                            context: [context_tokens for _ in range(self.batch_size)]
                        })
                        for i in range(self.batch_size):
                            generated += 1
                            text = enc.decode(out[i])

                        generated_word = text.split()[-1]
                        generated_words.append(generated_word)

                old_new_words.append([ind, words[ind], generated_words])


                #TODO WORD2VEC WORD SELECTION HERE!
                closest, closest_ind = [1000, 0]

                for candidates in generated_words:
                    try:
                        cur_similarity = embeddings.similarity(words[ind], candidates)
                        if cur_similarity > closest:
                            closest = cur_similarity
                            closest_ind = ind
                    except:
                        print('Unknown word!')
                loss += closest
                words[ind] = generated_words[closest_ind]
            try:
                print('='*40 + 'similarity: ' + loss/len(old_new_words) + ' ' + '='*40)
            except:
                print('Not enough data for similarity!')
            return [words, old_new_words]



logging.info('Building ReplaceableWordsDetector')
det = ReplaceableWordsDetector()
TextGenerator = interact_model()

def transform(text, return_mapping=False):
    logging.info('Called transform with text "' + text + '"')
    result = det.get_replaceable_words(text.translate(str.maketrans('', '', string.punctuation)))
    result, new_old_words = TextGenerator.generate(result[1], result[0])

    print(new_old_words)
    if return_mapping:
        return result, new_old_words
    else:
        indexes = [x[0] for x in new_old_words]
        new_words = (x[2] for x in new_old_words)
        #print(indexes, new_words)

        i = 0
        new_text = ''
        prev = ''
        c = ',.-—:;!?'
        for e in text + ' ':
            if e == ' ':
                if prev not in c:
                    if i in indexes:
                        new_text += next(new_words)
                    i += 1
                new_text += e
            elif e in c:
                if i in indexes:
                    new_text += next(new_words)
                i += 1
                new_text += e
            elif i not in indexes:
                new_text += e
            prev = e
            #print(e, i, new_text)

        return new_text[:-1]
