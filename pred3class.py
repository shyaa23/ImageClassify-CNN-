from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import cv2

# load trained model
json_file = open('model3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into model
classifier.load_weights("model3.h5")
print("Loaded the model")


# Making new predictions

test_image = image.load_img('pics/test/everest/everest-63.jpg', target_size = (64, 64))
#test_image = image.load_img('pics/test/kanchenjunga/kanchenjunga-88.jpg', target_size = (64, 64))
#test_image = image.load_img('pics/test/manaslu/manaslu-71.jpg', target_size = (64, 64))

img= cv2.imread('pics/test/everest/everest-63.jpg')
#img= cv2.imread('pics/test/kanchenjunga/kanchenjunga-88.jpg')
#img= cv2.imread('pics/test/manaslu/manaslu-71.jpg')
img = cv2.resize(img, (700,500), interpolation = cv2.INTER_CUBIC)

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
answer = np.argmax(result)

def label():
    if answer == 0:
        return "Predicted: Everest"

    elif answer == 1:
        return "Predicted: Kanchenjunga"

    elif answer == 2:
        return "Predicted: Manaslu"

cv2.putText(img, label(), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), lineType=cv2.LINE_AA)
cv2.imshow('Result',img)
cv2.waitKey(0)


'''
print(result)

#FOR SOFTMAX FUNCTION

if answer == 0:
    print("Label: Everest")

elif answer == 1:
    print("Labels: Kanchenjunga")

elif answer == 2:
    print("Label: Manaslu")
'''