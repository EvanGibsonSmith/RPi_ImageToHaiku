import tensorflow as tf
import numpy as np
import tf_keras.datasets.cifar100 as cf

cifar100_fine_labels = [

    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',

    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',

    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',

    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',

    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',

    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',

    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',

    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',

    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',

    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',

    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',

    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',

    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',

    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'

]

def start(path="model.tflite"):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter,input_details,output_details

def run(image_input=np.array([]),interpreter=None, input_details=None, output_details=None):
    interpreter.set_tensor(input_details[0]['index'], image_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index']) # np.array of shape (1, 100)
    return [cifar100_fine_labels[np.argsort(output_data[0])[-i]] for i in range(5)]

def test():
    interpreter,input_details,output_details = start()
    (x_train, y_train), (x_test, y_test) = cf.load_data()
    for idx in range(100):
        input_image =  np.array([x_test[idx]], dtype=np.float32)
        input_class = y_test[idx]
        output_data = run(image_input=input_image,interpreter=interpreter, input_details=input_details, output_details=output_details)
        print("Predicted: ", output_data)
        print("Truth:     ", cifar100_fine_labels[input_class[0]])

if __name__ == "__main__":
    test()
# Test model on random input data
#input_shape = input_details[0]['shape'] # [1, 32, 32, 3]
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
#interpreter.set_tensor(input_details[0]['index'], input_data)

#interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data
#output_data = interpreter.get_tensor(output_details[0]['index']) # np.array of shape (1, 100)
#print(output_data)
# 
# (x_train, y_train), (x_test, y_test) = cf.load_data()
# 
# idx = 0
# interpreter.set_tensor(input_details[0]['index'], np.array([x_train[idx]], dtype=np.float32))
# 
# interpreter.invoke()
# 
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)
# print(np.argmax(output_data))
# print(y_train[idx])