import cv2
import os, shutil
import handsegment as hs
import tensorflow.compat.v1 as tf
import numpy as np
from collections import deque
import tflearn

tf.disable_v2_behavior()

test_video = 'test_video/learn.mp4'
test_out_frames = 'test_out_frames'

def video_to_frames():
    if os.path.exists('test_out_frames'):
        shutil.rmtree('test_out_frames')
        
    os.mkdir('test_out_frames')
    cap = cv2.VideoCapture(test_video)
    lastFrame = None
    
    count = 0
    while count < 201:
        ret, frame = cap.read()
        
        if ret is False:
            break
        
        framename = test_out_frames + "/test_frame_" + str(count) + ".jpeg"
        frame = hs.handsegment(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lastFrame = frame
        cv2.imwrite(framename, frame)
        count += 1
    
    while count < 201:
        framename = test_out_frames + "/test_frame_" + str(count) + ".jpeg"
        cv2.imwrite(framename, lastFrame)
        count += 1
    
    cap.release()
    cv2.destroyAllWindows()


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def predict(graph, image_tensor, input_layer, output_layer):
    with tf.Session(graph=graph) as sess:
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)
        results = sess.run(
            output_operation.outputs[0],
            {input_operation.outputs[0]: image_tensor}
        )
    
    return results


def read_tensor_from_image_file(frames, input_height=299, input_width=299, input_mean=0, input_std=255):
    input_name = "file_reader"
    frames = [(tf.read_file(frame, input_name), frame) for frame in frames]
    
    decoded_frames = []
    
    for frame in frames:
        file_reader = frame[0]
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
        decoded_frames.append(image_reader)
    
    float_caster = [tf.cast(image_reader, tf.float32) for image_reader in decoded_frames]
    float_caster = tf.stack(float_caster)
    resized = tf.image.resize_bilinear(float_caster, [input_height, input_width])
    
    sess = tf.Session()
    with tf.Session() as sess:
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        result = sess.run(normalized)
        
    return result


def get_network_wide(frames, input_size, num_classes):
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 256, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', name='output1')
    return net

def load_labels(label_file):
    label = {}
    count = 0
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    
    for l in proto_as_ascii_lines:
        label[count] = l.strip()
        count += 1
    return label


def main():
    # Convert video into pieces of image frames
    video_to_frames()
    
    # Get normalized image tensor
    graph = load_graph('retrained_graph.pb')
    batch_size = 67
    preds = []
    batch = [test_out_frames + '/' + i for i in os.listdir(test_out_frames)]
    for i in [batch[i:i + batch_size] for i in range(0, len(batch), batch_size)]:
        input_height = 299
        input_width = 299
        input_mean = 0
        input_std = 255
        
        try:
            frames_tensors = read_tensor_from_image_file(
                i, 
                input_height=input_height, 
                input_width=input_width, 
                input_mean=input_mean, 
                input_std=input_std
            )
            
            pred = predict(graph, frames_tensors, 'Placeholder', 'final_result')
            pred = [[each.tolist(), ''] for each in pred]
            preds.extend(pred)
        except Exception as e:
            print(e)
    
    # Get features from each frame
    X = []
    temp_list = deque()
    num_frames_per_video = 201
    
    for i, frame in enumerate(preds):
        features = frame[0]
        
        # Add to the queue.
        if len(temp_list) == num_frames_per_video - 1:
            temp_list.append(features)
            flat = list(temp_list)
            X.append(np.array(flat))
            temp_list.clear()
        else:
            temp_list.append(features)
            continue
    
    X = np.array(X)
    print(X.shape)
        
    # Predict
    labels = load_labels('retrained_labels.txt')
    num_classes = 5
    size_of_each_frame = 5
    net = get_network_wide(num_frames_per_video, size_of_each_frame, num_classes)
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.load('checkpoints/' + 'non_pool.model')
    predictions = model.predict(X)
    predictions = np.array([np.argmax(pred) for pred in predictions])
    
    print(labels[predictions[0]])
    
if __name__ == '__main__':
    main()