
import math
import numpy as np
import os
import nltk
import json

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("caption_file", "data/mscoco/raw-data/annotations/captions_val2014.json",
                       "File containing the captions.")
tf.flags.DEFINE_string("id_list_file", "data/images_in_testset.txt",
                       "File containing ids of test images.")
tf.flags.DEFINE_string("file_path_prepend", "data/mscoco/raw-data/val2014/",
                       "Path to folder with test images.")
tf.flags.DEFINE_bool("verbose", False,
                     "Print every caption")
# Use Baseline or improved model
tf.flags.DEFINE_bool("improved", True,
                     "Use improved model")


tf.logging.set_verbosity(tf.logging.INFO)


def load_test_ids(test_id_file):
    ids = []
    with open(test_id_file, 'r') as f:
        for line in f:
            ids.append(int(line))
    return ids


def load_caption_data(caption_file, use_ids):
    # caption file: the json file containing the captions
    # use_ids: the ids which should be used
    with open(caption_file, 'r') as json_file:
        caption_data = json.load(json_file)
    id_to_filename = {}
    for x in caption_data["images"]:
        img_id = x["id"]
        if img_id in use_ids:
            id_to_filename[img_id] = FLAGS.file_path_prepend + x["file_name"]

    id_to_captions = {}
    for annotation in caption_data["annotations"]:
        image_id = annotation["image_id"]
        if image_id in use_ids:
            caption = annotation["caption"]
            id_to_captions.setdefault(image_id, [])
            id_to_captions[image_id].append(caption)

    return id_to_filename, id_to_captions



def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path,
                                               improved=FLAGS.improved)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  test_ids = load_test_ids(FLAGS.id_list_file)
  id_to_filename, id_to_captions = load_caption_data(FLAGS.caption_file, test_ids)

  print("Loaded %d ids" % len(test_ids))

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    global_bleu1_scores = []
    global_bleu2_scores = []
    global_bleu4_scores = []

    output_captions = []

    for t, img_id in enumerate(test_ids):
        filename = id_to_filename[img_id]
        with tf.gfile.GFile(filename, "rb") as f:
            image = f.read()
        captions = generator.beam_search(sess, image)
        if FLAGS.verbose:
            print("Captions for image %s:" % os.path.basename(filename))
        bleu1_scores = []
        bleu2_scores = []
        bleu4_scores = []
        first_sentence = " ".join([vocab.id_to_word(w) for w in captions[0].sentence[1:-1]])
        print(first_sentence)
        output_captions.append({"image_id": img_id, "caption": first_sentence})
        for i, caption in enumerate(captions):
            # Ignore begin and end words.
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            bleu1_scores.append(nltk.translate.bleu(id_to_captions[img_id], sentence, weights=[1.]))
            bleu2_scores.append(nltk.translate.bleu(id_to_captions[img_id], sentence, weights=[0.5, 0.5]))
            bleu4_scores.append(nltk.translate.bleu(id_to_captions[img_id], sentence, weights=[0.25, 0.25, 0.25, 0.25]))
            if FLAGS.verbose:
                sentence = " ".join(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                print("Bleu:", bleu4_scores[-1])
        global_bleu1_scores.append(max(bleu1_scores))
        global_bleu2_scores.append(max(bleu2_scores))
        global_bleu4_scores.append(max(bleu4_scores))
        if FLAGS.verbose:
            for caption in id_to_captions[img_id]:
                print(caption)
            print(np.mean(bleu1_scores))
            print(np.mean(bleu2_scores))
            print(np.mean(bleu4_scores))
        if t % 10 == 0:
            print("%d / %d" % (t,len(test_ids)))
            print(np.mean(global_bleu1_scores))
            print(np.mean(global_bleu2_scores))
            print(np.mean(global_bleu4_scores))
        if t > 100:
            break

    print("Mean Bleu-1, Bleu-2, Bleu-4 scores:")
    print(np.mean(global_bleu1_scores))
    print(np.mean(global_bleu2_scores))
    print(np.mean(global_bleu4_scores))

    with open('captions_val2014_im2txt_results.json', 'w') as outfile:
        json.dump(output_captions, outfile)


if __name__ == "__main__":
  tf.app.run()