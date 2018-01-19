import math
import os
import json

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "model/train/model.ckpt-250000",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "data/mscoco/word_counts.txt", "Text file containing the vocabulary.")
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
# use bulb or simple beam
tf.flags.DEFINE_bool("bulb", True,
                     "Use bulb search")
# set beam size
tf.flags.DEFINE_integer("beam",3,"beam size")

# varing beam size
tf.flags.DEFINE_bool("vary",False,"vary beam size")

# set break point
tf.flags.DEFINE_integer("bp",200,"break point")

#set normalization factor
tf.flags.DEFINE_float("nf",1.0,"length normalization factor")

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
    generator = caption_generator.CaptionGenerator(model, vocab, beam_size=FLAGS.beam, length_normalization_factor=FLAGS.nf)

    if FLAGS.bulb:
      out_file_path = 'captions_val2014_bulb'+str(FLAGS.beam)
    elif FLAGS.vary:
      out_file_path = 'captions_val2014_vary'+str(FLAGS.beam)
    else:
      out_file_path = 'captions_val2014_beam'+str(FLAGS.beam)
    out_file_path += '_'+str(FLAGS.nf)+'_results.json'
    try:
      with open(out_file_path, 'r') as infile:
        output_captions = json.load(infile)
    except Exception:
      output_captions = []
    for t, img_id in enumerate(test_ids):
        if t<len(output_captions):
          continue
        filename = id_to_filename[img_id]
        try:
          with tf.gfile.GFile(filename, "rb") as f:
            image = f.read()
        except Exception:
          print("skip id",img_id,filename)
          continue

        if FLAGS.bulb:
          captions = generator.bulb_beam_search(sess, image)
        elif FLAGS.vary:
          captions = generator.vary_beam_search(sess, image)
        else:
          captions = generator.beam_search(sess, image)

        first_sentence = " ".join([vocab.id_to_word(w) for w in captions[0].sentence[1:-1]])
        output_captions.append({"image_id": img_id, "caption": first_sentence})

        if FLAGS.verbose:
            print("Captions for image %s:" % os.path.basename(filename))
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

        if FLAGS.verbose:
            for caption in id_to_captions[img_id]:
                print(caption)

        if t % 10 == 0:
            print("%d / %d" % (t,len(test_ids)))
        if t == FLAGS.bp:
          break

    with open(out_file_path, 'w') as outfile:
        json.dump(output_captions, outfile)


if __name__ == "__main__":
  tf.app.run()
