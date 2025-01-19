from typing import List

class Perceptron:
    def __init__(self, num_inputs: int, learning_rate: float = 0.01) -> None:
        # Inicializa os pesos e o viés manualmente
        self.weights: List[float] = [0.0 for _ in range(num_inputs)]  # Pesos começam com zero
        self.bias: float = 0.0  # O viés começa com zero
        self.learning_rate: float = learning_rate  # Taxa de aprendizado

    def activation_function(self, z: float) -> int:
        # Função de ativação (step)
        return 1 if z > 0 else 0

    def predict(self, inputs: List[float]) -> int:
        # Calcula a soma ponderada: z = w1*x1 + w2*x2 + ... + wn*xn + viés
        weighted_sum = sum(weight * input_value for weight, input_value in zip(self.weights, inputs)) + self.bias
        return self.activation_function(weighted_sum)

    def train(self, training_data: List[List[float]], labels: List[int], epochs: int = 10) -> None:
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            for inputs, label in zip(training_data, labels):
                # Faz uma previsão
                prediction = self.predict(inputs)
                # Calcula o erro
                error = label - prediction
                # Atualiza os pesos e o viés
                for i in range(len(self.weights)):
                    self.weights[i] += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error
                # Log para depuração
                print(f"Inputs: {inputs}, Prediction: {prediction}, Error: {error}, Weights: {self.weights}, Bias: {self.bias}")
            print()

if __name__ == "__main__":
    # Dados de treinamento (OR lógico)
    training_data: List[List[float]] = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    labels: List[int] = [0, 1, 1, 1]  # Saídas esperadas

    # Cria o perceptron
    perceptron = Perceptron(num_inputs=2, learning_rate=0.1)

    # Treina o perceptron
    perceptron.train(training_data, labels, epochs=10)

    # Testa o perceptron
    print("Testing the perceptron:")
    for inputs in training_data:
        print(f"Input: {inputs}, Prediction: {perceptron.predict(inputs)}")
