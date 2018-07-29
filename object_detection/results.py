import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

sys.path.append("..")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')





from utils import label_map_util

from utils import visualization_utils as vis_util





MODEL_NAME = 'results'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'


PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

NUM_CLASSES = 1




detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



PATH_TO_TEST_IMAGES_DIR = 'images/test'
TEST_IMAGE_PATHS=[]

for files in sorted(os.listdir('images/test')):
    if(files.split('.')[1]!='xml'):
        TEST_IMAGE_PATHS.append( os.path.join(PATH_TO_TEST_IMAGES_DIR,files))





IMAGE_SIZE = (12, 8)



def CheckResults(names,scores,boxes,image):
    print (names)

    tree = ET.parse('images/test/'+names.split('.')[0]+'.xml')
    root = tree.getroot()
    XMIN=[]
    YMIN=[]
    XMAX=[]
    YMAX=[]

    for rank in root.iter('width'):
        width=(int(rank.text))
    for rank in root.iter('height'):
        height=(int(rank.text))

    for rank in root.iter('xmin'):
        XMIN.append(int(rank.text))
    for rank in root.iter('ymin'):
        YMIN.append(int(rank.text))
    for rank in root.iter('xmax'):
        XMAX.append(int(rank.text))
    for rank in root.iter('ymax'):
        YMAX.append(int(rank.text))
    mids=[]
    for idx,vals in enumerate(XMIN):
        mids.append([(XMIN[idx]+XMAX[idx])/2.0,(YMIN[idx]+YMAX[idx])/2.0])
    TP=0.0
    FP=0.0
    res=[]
    for idx,scor in enumerate(scores):
        c=0.0
        if (scor >=0.5):
            ymin=(boxes[idx][0])*height
            xmin=(boxes[idx][1])*width
            ymax=(boxes[idx][2])*height
            xmax=(boxes[idx][3])*width

            for vals in mids:
                if(vals[0]>=xmin and vals[0] <=xmax and  vals[1]>=ymin and vals[1]<=ymax):
                    TP=TP+1.0

                    d=np.sqrt(np.square(xmax-((xmin+xmax)/2.0))+ np.square(ymin-((ymin+ymax)/2.0)))*(3/4)
                    cv2.circle(image, ( int((xmin+xmax)/2.0),int((ymin+ymax)/2.0)),int(d)  , (0,255,0), 5)
                    mids.remove(vals)
                    c=1.0

            if(c==0.0):
                FP=FP+1.0
                d=np.sqrt(np.square(xmax-((xmin+xmax)/2.0))+ np.square(ymin-((ymin+ymax)/2.0)))*(3/4)
                cv2.circle(image, ( int((xmin+xmax)/2.0),int((ymin+ymax)/2.0)),int(d)  , (255,0,0), 5)


    '''
    for idx,vals in enumerate(XMIN):
        meh=[(XMIN[idx]+XMAX[idx])/2.0,(YMIN[idx]+YMAX[idx])/2.0]
        for mid in mids:
            if(mid==meh):
                d=np.sqrt(np.square(XMAX[idx]-((XMIN[idx]+XMAX[idx])/2.0))+ np.square(YMIN[idx]-((YMIN[idx]+YMAX[idx])/2.0)))*(3/4)
                cv2.circle(image, ( int((XMIN[idx]+XMAX[idx])/2.0),int((YMIN[idx]+YMAX[idx])/2.0)),int(d)  , (0,0,255), 5)
    '''


    FN=len(mids)+0.0
    print(TP,FP,FN)

    '''
    plt.imshow(image)
    plt.show()

    '''
    return TP,FP,FN






def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:

      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:

        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)

        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict



TP=[]
FP=[]
FN=[]
for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  image_np = load_image_into_numpy_array(image)
  image_np_expanded = np.expand_dims(image_np, axis=0)
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  '''
  tp,fp,fn=CheckResults(image_path.split('/')[-1],output_dict['detection_scores'],output_dict['detection_boxes'],image_np)
  TP.append(tp)
  FP.append(fp)
  FN.append(fn)
  '''


  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image_np)
  plt.show()
'''


print (TP)
print(FP)
print (FN)
print ('Sensitivity/Recall:',(sum(TP))/(sum(TP)+sum(FN)))
print ('Precision:',(sum(TP))/(sum(TP)+sum(FP)))
print ('F1 score:',(2*sum(TP))/(2*sum(TP)+sum(FN)+sum(FP)))
'''
