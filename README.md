# Show and Tell: A Neural Image Caption Generator


https://github.com/tensorflow/models/tree/master/research/im2txt

Made compatible with python 3.

Changes include:
In general, change tf.gfile.GFile(filename, "r") to tf.gfile.GFile(filename, "rb") because python3 treats "r" as strings instead of bytes.

For the preprocessing:
In the _bytes_feature_list function use:
_bytes_feature(bytes(v,encoding='utf-8')) for v in values
and remove the cast str of the value in the _bytes_feature(value) function.