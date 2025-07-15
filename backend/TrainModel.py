from FaceModel import FaceModel
import numpy as np
from CreateDataset import create_dataset

data, test_data, labels, testLabels = create_dataset("Dataset", True)
negData, test_neg_data, negLabels, test_negLabels = create_dataset("DatasetNegative", False)
data = np.concatenate([data, negData], axis=0)
labels = np.concatenate([labels, negLabels], axis=0)
test_data = np.concatenate([test_data, test_neg_data], axis=0)
testLabels = np.concatenate([testLabels, test_negLabels], axis=0)
model = FaceModel(data, len(labels), 512 * 512, labels)
model.train(1000)

model.save()
