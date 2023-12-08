from model-training import X_test, y_test
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


model = load_model('ship_detection.h5')


'''Evaluating the Model'''
predictions = model.predict_classes(X_test)
print(classification_report(y_test, predictions))
print(model.metrics_names)
print(model.evaluate(X_test,y_test,verbose=0))


'''Predicting a given image'''
my_image = X_test[18]
answer = model.predict_classes(my_image.reshape(1,768,768,3))
print(answer)     # its 1 as this sample contains ship
