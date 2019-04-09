from keras.models import load_model
from keras import backend as K


sess = K.get_session()

yolo_model = load_model("model_data/trained_weights_stage_1.h5")

yolo_model.summary()

