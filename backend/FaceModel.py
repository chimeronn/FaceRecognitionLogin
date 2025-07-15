import numpy as np

class FaceModel:
    def __init__(self, data, num_samples, image_size, labels):
        self.W = np.random.randn(image_size)
        self.b = np.random.randn()
        self.X = data
        self.labels = labels
        self.num_samples = num_samples

    def sigmoid(self, logits):
        return 1 / (1 + np.exp(-np.array(logits)))
    
    def cross_entropy_loss(self, pred):
        pred = np.clip(pred, 1e-10, 1 - 1e-10)
        return -np.sum(self.labels * np.log(pred)) / self.num_samples
    
    def step(self):
        logits = np.dot(self.W, self.X.T) + self.b
        pred = self.sigmoid(logits)
        grad_W = np.dot(self.X.T, (pred - self.labels)) / self.num_samples
        grad_b = np.mean(pred - self.labels)
        return grad_W, grad_b

    def update(self, grad_W, grad_b, learning_rate = 0.01):
        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b

    def train(self, num_epochs=3000):
        for epoch in range(num_epochs):
            grad_W, grad_b = self.step()
            self.update(grad_W, grad_b)
            if epoch % 100 == 0:
                logits = np.dot(self.W, self.X.T) + self.b
                pred = self.sigmoid(logits)
                loss = self.cross_entropy_loss(pred)
                print(f"Iteration {epoch + 1}, Loss: {loss}")
    
    def predict(self, image):
        logits = np.dot(image, self.W) + self.b
        pred = self.sigmoid(logits)
        return pred >= 0.5
    
    def save(self, path="face_model.npz"):
        np.savez(path, W=self.W, b=self.b)

    def load(self, path="face_model.npz"):
        data = np.load(path)
        self.W = data["W"]
        self.b = data["b"]