import math
import random

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NN:
    def __init__(self, ni, nh, no, learn=0.05, correct=0.1):
        self.input_num = ni + 1  # add one for bias neuron
        self.hidden_num = nh
        self.output_num = no

        self.input_cells = [1.0] * self.input_num
        self.hidden_cells = [1.0] * self.hidden_num
        self.output_cells = [1.0] * self.output_num

        self.input_weights = make_matrix(self.input_num, self.hidden_num)
        self.hidden_weights = make_matrix(self.hidden_num, self.output_num)

        for i in range(self.input_num):
            for j in range(self.hidden_num):
                self.input_weights[i][j] = rand(-1.0, 1.0)

        for i in range(self.hidden_num):
            for j in range(self.output_num):
                self.hidden_weights[i][j] = rand(-1.0, 1.0)

        self.input_correction = make_matrix(self.input_num, self.hidden_num)
        self.hidden_correction = make_matrix(self.hidden_num, self.output_num)

        self.learn = learn
        self.correct = correct

    def forward(self, inputs):
        # initialize input neurons
        for i in range(self.input_num - 1):
            self.input_cells[i] = inputs[i]

        # activate input layer
        for i in range(self.hidden_num):
            val = 0
            for j in range(self.input_num):
                val += self.input_cells[j] * self.input_weights[j][i]
            self.hidden_cells[i] = sigmoid(val)

        # activate output layer
        for i in range(self.output_num):
            val = 0
            for j in range(self.hidden_num):
                val += self.hidden_cells[j] * self.hidden_weights[j][i]
            self.output_cells[i] = sigmoid(val)

        return self.output_cells[:]

    def backward(self, inputs, label):
        self.forward(inputs)

        output_diff = [0.0] * self.output_num
        for i in range(self.output_num):
            error = label[i] - self.output_cells[i]
            output_diff[i] = sigmoid_derivative(self.output_cells[i]) * error

        hidden_diff = [0.0] * self.hidden_num
        for i in range(self.hidden_num):
            error = 0.0
            for j in range(self.output_num):
                error += output_diff[j] * self.hidden_weights[i][j]
            hidden_diff[i] = sigmoid_derivative(self.hidden_cells[i]) * error

        # print(self.correct)
        # update weights
        for i in range(self.hidden_num):
            for j in range(self.output_num):
                change = output_diff[j] * self.hidden_cells[i]
                self.hidden_weights[i][j] += self.learn * change + self.correct * self.hidden_correction[i][j]
                self.hidden_correction[i][j] = change

        for i in range(self.input_num):
            for j in range(self.hidden_num):
                change = hidden_diff[j] * self.input_cells[i]

                self.input_weights[i][j] += self.learn * change + self.correct * self.input_correction[i][j]
                self.input_correction[i][j] = change

        error = 0.0
        for i in range(len(label)):
            error += 0.5 * (label[i] - self.output_cells[i]) ** 2
        return error

    def train(self, inputs, labels, limit=10000, learn=0.05, correct=0.1):
        self.learn = learn
        self.correct = correct

        for j in range(limit):
            error = 0.0
            for i in range(len(inputs)):
                label = labels[i]
                input = inputs[i]
                error += self.backward(input, label)
            if error < 0.005:
                print(j)
                break

    def test(self, name):

        file = open(name, 'r')

        inputs = []
        outputs = []

        for line in file:
            line = line.replace("\n", "")
            if len(line) == 0:
                continue

            line = line.split(",")

            for i in range(len(line)):
                try:
                    line[i] = float(line[i])
                except:
                    pass

            inputs.append(line[0: len(line) - 1])
            outputs.append(line[len(line) - 1])

        file.close()

        # build label dictionary
        label_set = set(outputs)
        label_dict = {}

        id = 0
        for i in label_set:
            val = [0] * len(label_set)
            val[id] = 1
            label_dict[i] = val
            id = id + 1

        # split into training set and testing set
        data_num = len(inputs)

        train_id = random.sample(range(0, data_num - 1), int(data_num * 2 / 3))

        train_cases = []
        train_labels = []
        test_cases = []
        test_labels = []

        for i in range(data_num):
            if i in train_id:
                train_cases.append(inputs[i])
                train_labels.append(label_dict[outputs[i]])
            else:
                test_cases.append(inputs[i])
                test_labels.append(label_dict[outputs[i]])

        self.train(train_cases, train_labels, 10000, 0.05, 0.1)

        for i in range(len(test_cases)):
            print(self.forward(test_cases[i]))
            print(test_labels[i])


if __name__ == '__main__':
    nn = NN(4, 5, 3, 0.05, 0.1)
    name = "iris.txt"
    nn.test(name)
