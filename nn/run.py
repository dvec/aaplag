import json
import os
import numpy as np
import tensorflow as tf
import regex as re

# import sys
# sys.modules['model'] = __import__('src.model')
# sys.modules['sample'] = __import__('src.sample')
# sys.modules['encoder'] = __import__('src.encoder')

# from src import model, sample, encoder
import nn.model as model
import nn.sample as sample
import nn.encoder as encoder


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
        ratio=0.1,
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
            elif not isFitForReplacement(words[ind], ind, keyword_indicies):
                continue
            else:
                raw_text = join_words(words, ind)
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
            old_new_words.append([words[ind], ])
            words[ind] = text.split()[-1]
        print([words, old_new_words])
        # return [words, old_new_words]


X = [[32, 96, 68, 6, 8, 21, 25, 59, 95, 57],
     ['Quite',
      'literally',
      ',',
      'the',
      'term',
      '``',
      'philosophy',
      "''",
      'means',
      ',',
      '``',
      'love',
      'of',
      'wisdom',
      '.',
      "''",
      'In',
      'a',
      'broad',
      'sense',
      ',',
      'philosophy',
      'is',
      'an',
      'activity',
      'people',
      'undertake',
      'when',
      'they',
      'seek',
      'to',
      'understand',
      'fundamental',
      'truths',
      'about',
      'themselves',
      ',',
      'the',
      'world',
      'in',
      'which',
      'they',
      'live',
      ',',
      'and',
      'their',
      'relationships',
      'to',
      'the',
      'world',
      'and',
      'to',
      'each',
      'other',
      '.',
      'As',
      'an',
      'academic',
      'discipline',
      'philosophy',
      'is',
      'much',
      'the',
      'same',
      '.',
      'Those',
      'who',
      'study',
      'philosophy',
      'are',
      'perpetually',
      'engaged',
      'in',
      'asking',
      ',',
      'answering',
      ',',
      'and',
      'arguing',
      'for',
      'their',
      'answers',
      'to',
      'lifeâ€™s',
      'most',
      'basic',
      'questions',
      '.',
      'To',
      'make',
      'such',
      'a',
      'pursuit',
      'more',
      'systematic',
      'academic',
      'philosophy',
      'is',
      'traditionally',
      'divided',
      'into',
      'major',
      'areas',
      'of',
      'study',
      '.']]
interact_model(X[1], X[0])
