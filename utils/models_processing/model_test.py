"""
    -> Model testing script.
    -> Input:
        - Path to the input test image directory.
        - Path to the frozen graph(.pb) file.
        - Path to the output directory
            
    -> command to run:
        python model_testing.py\
          --input_dir=<PATH_TO_THE_TEST_DIRECTORY>\
          --output_dir=<PATH TO THE OUTPUT DIRECTORY>\
          --model_file=<PATH_TO_THE_MODEL_FILE>\
          --label_file=<PATH TO THE LABEL FILE>\
          --threshold=<Threshold value for inference default is 0.5>
    -> Output:
        - Image file having rectangles drawn on it. 

"""
# package importing
import argparse
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
	with graph.as_default():
		with tf.Session() as sess:
			# Get handles to input and output tensors
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
				# The following processing is only for single image
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
				# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					detection_masks, detection_boxes, image.shape[0], image.shape[1])
				detection_masks_reframed = tf.cast(
					tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				# Follow the convention by adding back the batch dimension
				tensor_dict['detection_masks'] = tf.expand_dims(
					detection_masks_reframed, 0)
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			# Run inference
			output_dict = sess.run(tensor_dict,
									feed_dict={image_tensor: np.expand_dims(image, 0)})

			# all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict[
				'detection_classes'][0].astype(np.uint8)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]
	return output_dict

# load and return model.
def get_detection_graph(path_to_frozen_graph):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# entrypoint
if __name__ == "__main__":

	# command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", help="path to the test image directory",
						type=str)
	parser.add_argument("--output_dir", help="path to the output directory",
						type=str)
	parser.add_argument("--model_file", help="path to the frozen grapg file / model file (.pb)on which testing needs to perform",
						type=str)
	parser.add_argument("--label_file", help="path to the label file",
						type=str)
	parser.add_argument("--threshold", help="Threshold value for inference",
						type=float, default=0.5)

	args = vars(parser.parse_args())

    
	category_index = label_map_util.create_category_index_from_labelmap(args['label_file'], use_display_name=True)
	detection_graph = get_detection_graph(args['model_file'])
    
	for image_file in os.listdir(args['input_dir']):
		image_path = os.path.join(args['input_dir'], image_file)
		image = Image.open(image_path)
		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		image_np = load_image_into_numpy_array(image)
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		# Actual detection.
		output_dict = run_inference_for_single_image(image_np, detection_graph)
		# Visualization of the results of a detection.
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			category_index,
			instance_masks=output_dict.get('detection_masks'),
			use_normalized_coordinates=True,
			line_thickness=3,
			min_score_thresh=args['threshold'])
		
		im = Image.fromarray(image_np)
		im.save(os.path.join(args['output_dir'], image_file))
		print('Processed:', image_file)

 