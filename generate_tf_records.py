#!/usr/bin/python3
import argparse
import sys

ps = argparse.ArgumentParser(description='')
ps.add_argument("opath", type=str, help='Output path for models')
ps.add_argument("lpath", type=str, help='Labels path (.pbtxt)')
ps.add_argument("tjson", type=str, help='Training json')
ps.add_argument("tbbjson", type=str, help='Bounding boxes training json')
ps.add_argument("vjson", type=str, help='Validation json')
ps.add_argument("vbbjson", type=str, help='Validation bounding boxes json')
ps.add_argument("mjson", type=str, help='Root json')

args=ps.parse_args()

sys.path.append('/usr/xtmp/wjs/BirdTrain/models/research/')
sys.path.append('/usr/xtmp/wjs/BirdTrain/models/research/object_detection/')
sys.path.append('/usr/xtmp/wjs/BirdTrain/models/research/slim/')

from utils import dataset_util
from dataset_tools import tf_record_creation_util

import contextlib
import tensorflow as tf
import logging
import json
import os
import re
from PIL import Image

BASE_DIR = args.opath

def combine_dicts(images, boxes):
    entries = []
    category_ids = {}
    annotations = {i['image_id']: i for i in images['annotations']}
    img = {i['id']: i for i in images['images']}
    categories = {i['id'] + 1: i for i in images['categories']}
    # Internally renumber category_ids so they start at one instead of zero as zero is for background
    for category_id in categories:
        categories[category_id]['id'] += 1
    bounding_boxes = {i['image_id']: i for i in boxes['annotations']}
    for bb in bounding_boxes:
        bounding_boxes[bb]['category_id'] += 1
    for e in annotations:
        if annotations[e]['image_id'] in bounding_boxes and categories[annotations[e]['category_id'] + 1]['supercategory'] == 'Aves':
            annotations[e]['category_id'] += 1
            d = {**annotations[e], **img[e], **categories[annotations[e]['category_id']], **bounding_boxes[annotations[e]['image_id']]}
            entries.append(d)
    return entries

def create_tf_example(entry):
    height = entry['height']  # Image height
    width = entry['width']  # Image width
    filename = entry['file_name'].encode()  # Filename of the image. Empty if image is not from file
    image_format = b'jpeg'  # b'jpeg' or b'png'
    encoded_image_data = open(BASE_DIR + filename.decode('ascii'), 'rb').read()  # Encoded image bytes
    xmins = [float(entry['bbox'][0]/width)]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [float((entry['bbox'][0] + entry['bbox'][2])/width)]  # List of normalized right x coordinates in bounding box # (1 per box)
    ymins = [float(entry['bbox'][1]/height)]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [float((entry['bbox'][1] + entry['bbox'][3])/height)]  # List of normalized bottom y coordinates in bounding box # (1 per box)
    classes_text = [entry['name'].encode()]  # List of string class name of bounding box (1 per box)
    classes = [entry['category_id']]  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_record(output_filename, examples):
    """Creates a TFRecord file from examples.
    Args:
    output_filename: Path to where output file is saved.
    examples: Examples to parse and save to tf record.
    """
    count = 0
    writer = tf.io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        tf_example = create_tf_example(example)
        if tf_example:
            writer.write(tf_example.SerializeToString())
            count += 1
    writer.close()
    print("Records written:", count)

def gen_labels(entries):
    bfile = open(args.lpath, 'w', encoding='utf-8')
    categories = {i['category_id']: i['name'] for i in entries}
    for i in categories:
        class_entry = 'item {\n'
        class_entry += '  id: ' + str(i) + ',\n'
        class_entry += '  name: "' + categories[i] + '"\n'
        class_entry += '}' + '\n'
        bfile.write(class_entry)
    jfile = open(args.mjson, 'w')
    jfile.write(json.dumps(categories, default=str, indent=2))


train_images = json.loads(open(args.tjson, 'r').read())
train_bb = json.loads(open(args.tbbjson, 'r').read())
all_train_entries = combine_dicts(train_images, train_bb)
gen_labels(all_train_entries)

eval_images = json.loads(open(vjson, 'r').read())
eval_bb = json.loads(open(vbbjson, 'r').read())
all_eval_entries = combine_dicts(eval_images, eval_bb)

create_tf_record(args.opath+'/birds_inat_tfrecords_train.record', all_train_entries)
create_tf_record(args.opath+'/birds_inat_tfrecords_eval.record', all_eval_entries)
