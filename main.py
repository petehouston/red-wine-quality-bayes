import csv
import math
import random


def load_csv(filename):
    dataset = list()
    with open(filename) as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        for row in rows:
            dataset.append([float(x) for x in row])
    return dataset


def split_dataset(dataset, split_ratio, is_random=False):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    if is_random:
        copy = list(dataset)
        while len(train_set) < train_size:
            index = random.randrange(len(copy))
            train_set.append(copy.pop(index))
        return [train_set, copy]
    return [dataset[:train_size], dataset[train_size:]]


def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    # print(separated)
    return separated


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculate_probability(x, x_mean, x_stdev):
    exponent = math.exp(-(math.pow(x - x_mean, 2) / (2 * math.pow(x_stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * x_stdev)) * exponent


def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            xmean, xstdev = classSummaries[i]
            x = input_vector[i]
            probabilities[classValue] *= calculate_probability(x, xmean, xstdev)
    return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for classValue, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = classValue
    return best_label


def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        # print('Test Set: {0}'.format(test_set[x]), sep='')
        # print('Prediction: {0}'.format(predictions[x]))
        if test_set[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main():
    filename = 'red-wine-quality.csv'
    split_ratio = 0.67
    dataset = load_csv(filename)
    training_set, test_set = split_dataset(dataset, split_ratio, False)
    print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset), len(training_set), len(test_set)))
    # prepare model
    model = summarize_by_class(training_set)
    # test model
    predictions = get_predictions(model, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}'.format(accuracy))


if __name__ == '__main__':
    main()

