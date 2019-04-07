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

from deeppavlov import configs, build_model


class ReplaceableWordsDetector:
    ner_model = None

    def __init__(self):
        # NER model initner_rus
        self.ner_model = build_model(configs.ner.ner_ontonotes, download=True)

    def get_keywords_indicies(self, keywords, text):
        keywords_indicies = []
        lowered_text = [word.lower() for word in text]
        for ind, word in enumerate(lowered_text):
            if word in keywords:
                keywords_indicies.append(ind)
        return keywords_indicies

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
        indicies_keywords = self.get_keywords_indicies(keywords(text, split=True), result[0][0])

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


def interact_model(
        words,
        keyword_indicies,
        ratio=0.2,
        model_name='117M',
        seed=None,
        nsamples=1,
        batch_size=None,
        length=1,
        temperature=1,
        top_k=0,
):
    # MAGIC STUFF BEGIN!

    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError(f"can't get samples longer than window size: {hparams.n_ctx}")

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )[:, 1:]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        # MAGIC STUFF OVER

        old_new_words = []
        words = words.copy()

        for ind in range(len(words)):
            if ind < 10:
                continue
            elif not isFitForReplacement(words[ind], ind, keyword_indicies) or random.random() > ratio:
                continue
            else:
                raw_text = join_words(words, ind)
                # print(raw_text)
                context_tokens = enc.encode(raw_text)
                generated = 0
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })
                    for i in range(batch_size):
                        generated += 1
                        text = enc.decode(out[i])
            generated_word = text.split()[-1]
            old_new_words.append([ind, words[ind], generated_word])
            words[ind] = generated_word
        return [words, old_new_words]


logging.info('Building ReplaceableWordsDetector')
det = ReplaceableWordsDetector()


def transform(text, return_mapping=False):
    logging.info('Called transform with text "' + text + '"')
    result = det.get_replaceable_words(text.translate(str.maketrans('', '', string.punctuation)))
    result, new_old_words = interact_model(result[1], result[0])

    print(new_old_words)
    if return_mapping:
        return result, new_old_words
    else:
        indexes = (x[0] for x in new_old_words)
        new_words = (x[1] for x in new_old_words)

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
            print(e, new_text)

        return new_text[:-1]
