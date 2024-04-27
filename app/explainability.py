import sys
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import keras
from PIL.ImageQt import ImageQt 
from lime import lime_image
from tensorflow.keras.preprocessing import image
from skimage.segmentation import mark_boundaries
from PyQt6.QtWidgets import *
from PyQt6.QtGui import  *
from PyQt6.QtCore import *

imgWIDTH = 300
imgHEIGHT = 300
# load the model when first starting up
cnnModel = tf.keras.models.load_model('catdog.keras')

class App(QDialog):
    def __init__(self, parent = None):
        super(App, self).__init__(parent)
        self.title = 'Explainability AI App'

        # add all the potential applications here
        self.radioGroupBox()
        self.uploadButtonBox()
        self.imgLabelBox()
        self.predictionBox()

        # main layout
        mainLayout = QGridLayout()
        mainLayout.addLayout(self.radioBox, 0, 0)
        mainLayout.addLayout(self.uploadBox, 1, 0)
        mainLayout.addLayout(self.labelBox, 2, 0)
        mainLayout.addLayout(self.predBox, 3, 0)
        self.setLayout(mainLayout)

        # initialize app
        self.initUI()

    # initializes the UI
    def initUI(self):
        self.setWindowTitle(self.title)
        
        self.show()
    

    # adds the radio button group that chooses the type of explainability
    def radioGroupBox(self):
        self.radioBox = QHBoxLayout()
        self.radio1 = QRadioButton('LIME/Boundary Lines', self)
        self.radio2 = QRadioButton('GRAD-CAM/Heatmaps', self)
        self.radioGroup = QButtonGroup()
        self.radioBox.addWidget(self.radio1)
        self.radioBox.addWidget(self.radio2)
        self.radio1.setChecked(True)


    # add the box with upload button:
    def uploadButtonBox(self):
        self.uploadBox = QHBoxLayout()
        self.uploadBtn = QPushButton("upload image of Cat or Dog here")
        self.uploadBtn.clicked.connect(self.getFile)
        self.uploadBox.addWidget(self.uploadBtn)

    # function to upload file
    def getFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "c:\\", "Image Files (*.png *.jpg)")
        # if there is a file, send it to the model to be predicted
        if fname != '':
            self.imgLabel.setText(fname)
            pred, npArr, npConvert = self.predictIMG(str(fname))
            if npConvert:
                h, w, c = npArr.shape
                bytesPerLine = 3 * w
                npArr = (npArr*255).astype('uint8')
                qImg = QImage(npArr, w, h, bytesPerLine, QImage.Format.Format_RGB888)
                pixIMG = QPixmap(qImg)
                self.imgLabel.setPixmap(pixIMG)
            else:
                print(type(npArr))
                qim = ImageQt(npArr)
                pixIMG = QPixmap.fromImage(qim)
                self.imgLabel.setPixmap(pixIMG)
            predConfi = ''
            if pred[0][0] > 0.5:
                predConfi += str(pred[0][0]*100) + "% confidence for DOG"
            else:
                predConfi += str((1-pred[0][0])*100) + "% confidence for CAT"
            self.predLabel.setText(predConfi)
        
    # label to upload images to
    def imgLabelBox(self):
        self.labelBox = QHBoxLayout()
        self.imgLabel = QLabel('Explained Image will appear here',self)
        # self.imgLabel.setFixedSize(imgWIDTH, imgHEIGHT)
        self.labelBox.addWidget(self.imgLabel)

    # label to upload predictions and confidence
    def predictionBox(self):
        self.predBox = QHBoxLayout()
        self.predLabel = QLabel('Prediction: ', self)
        self.predBox.addWidget(self.predLabel)

    # function to get the requested image, get the prediction, explain with the requested explainer
    def predictIMG(self, fPath):
        preprocessed_image = self.cnn_preprocess_image(fPath)
        pred = self.cnn_predict(cnnModel, preprocessed_image)
        if self.radio1.isChecked():
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                image=preprocessed_image[0],
                classifier_fn=lambda x: self.cnn_predict(cnnModel, x),
                top_labels=2,
                hide_color=0,
                num_samples=1000
            )

            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=False,
                num_features=5,
                hide_rest=False
            )
            limeIMG = mark_boundaries(temp / 2 + 0.5, mask)
            return pred, limeIMG, True

        elif self.radio2.isChecked():
            heatmap = self.make_gradcam_heatmap(preprocessed_image, cnnModel, 'conv2d_19')
            gradIMG = self.save_and_display_gradcam(fPath, heatmap)
            # print(gradIMG)
            # print(type(gradIMG))
            return pred, gradIMG, False
            
    # preprocess uploaded images for prediction
    def cnn_preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        imgArr = image.img_to_array(img)
        imgArr = np.expand_dims(imgArr, axis=0)
        imgArr /= 255.  # Scale image pixels to 0-1
        return imgArr
    
    # predict with uploaded image
    def cnn_predict(self, model, img_array):
        return model.predict(img_array)

    #####################################################################
    # FROM Team, K. (n.d.). Keras documentation: Grad-CAM class activation 
    # visualization. https://keras.io/examples/vision/grad_cam/  
    #####################################################################

    def get_img_array(img_path, size):
        # `img` is a PIL image of size 299x299
        img = keras.utils.load_img(img_path, target_size=size)
        # `array` is a float32 Numpy array of shape (299, 299, 3)
        array = keras.utils.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 299, 299, 3)
        array = np.expand_dims(array, axis=0)
        return array


    def make_gradcam_heatmap(self,img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def save_and_display_gradcam(self, img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
        # Load the original image
        img = keras.utils.load_img(img_path)
        img = keras.utils.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = mpl.colormaps["jet"]

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = (jet_heatmap * alpha + img)
        superimposed_img = keras.utils.array_to_img(superimposed_img)
        return superimposed_img

# main
def main():
   app = QApplication(sys.argv)
   ex = App()
   ex.show()
   sys.exit(app.exec())
	
if __name__ == '__main__':
   main()