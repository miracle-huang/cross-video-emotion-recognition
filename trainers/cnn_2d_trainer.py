import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from trainers.base_trainer import BaseTrainer
import config

class CnnTwoDimensionTrainer(BaseTrainer):
    def __init__(self, train_data, val_data, test_data, model, emotion_type):
        '''
        model - One trainer class only train one model in once time
        '''
        super().__init__(train_data, val_data, test_data)
        self.model = model

        # Check the value of emotion_type
        if emotion_type not in ['arousal', 'valence']:
            raise ValueError("emotion_type must be either 'arousal' or 'valence'")
        self.emotion_type = emotion_type

    def train_model(self):
        x_train = self.train_data['x_train'] # Total training data
        y_a_train = self.train_data['y_a_train'] # Total training arousal label
        y_v_train = self.train_data['y_v_train'] # Total training valence label

        x_val = self.val_data['x_val'] # validation data
        y_a_val = self.val_data['y_a_val'] # validation arousal label
        y_v_val = self.val_data['y_v_val'] # validation valence label   

        x_test = self.test_data['x_test'] # test data
        y_a_test = self.test_data['y_a_test'] # test arousal label
        y_v_test = self.test_data['y_v_test'] # test valence label

        # select which kind of label by emotion type
        y_train = y_val = y_test = None # train, validation, test type
        if self.emotion_type == "valence":
            y_train = y_v_train
            y_val = y_v_val
            y_test = y_v_test
        elif self.emotion_type == "arousal":
            y_train = y_a_train
            y_val = y_a_val
            y_test = y_a_test

        self.model.fit([x_train], # train data, Generates a list where each element is a column of x_train (feature)
            y_train,
            validation_data=([x_val],
                        y_val),
            # sample_weight=sample_weights,
            epochs=config.epoch,
            batch_size=config.batch_size,
            callbacks=self.callbacks,
            )
        
        # get result
        result = self.model.evaluate([x_test], y_test)
        result_output = f"{self.emotion_type} accuracy: {result[1]}"
        acc_output = result[1]
        print(result_output)

        # compute confusion matrix and f1 score
        predictions = self.model.predict([x_test])
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        conf_matrix = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        return acc_output, conf_matrix, f1
