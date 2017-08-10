from csv import reader
from operator import itemgetter
from math import log
import random

def readData(file):
    file = open(file, "r")
    data = []
    csvReader = reader(file, delimiter=",")
    csvReader.__next__()
    for row in csvReader:
        data.append((row[0:4], row[4]))
    return data


class NaiveBayesClassifier:
    def __init__(self):
        # Klassenvariablen für Klassen- und Featurewahrscheinlichkeiten
        self.ProbClasses = {}
        self.Features = {}
        self.ProbFeatures = {}
        self.vocabSize = 0

    def train(self, labeled_trainingset):
        """labeled_trainingset: Liste aus Tupeln, Tupeln bestehen aus Liste der Features und der Klasse
        Bsp:[([Feature1, Feature2,..],Klasse1),([Feature1, Feature2,..],Klasse2)...]"""
        self.calculateClassProbabilitiesAndSizeOfVocab(labeled_trainingset)
        self.calculateFeatureProbabilities(labeled_trainingset)

    # Methode zur Berechnung der Klassenwahrscheinlichkeiten
    # Zunächst werden die absoluten Häufigkeiten der Klassen im trainingsset bestimmt.
    # Anschließend wird die relative Häufigkeit der Klassen berechnet.
    def calculateClassProbabilitiesAndSizeOfVocab(self, labeled_trainingset):
        vocab = []
        for entry in labeled_trainingset:

            if entry[1] not in self.Features:
                self.Features[entry[1]] = []
            self.Features[entry[1]].extend(entry[0])

            if entry[1] not in self.ProbClasses:
                self.ProbClasses[entry[1]] = 1
            else:
                self.ProbClasses[entry[1]] += 1
            vocab.extend(entry[0])

        self.vocabSize = len(set(vocab))
        for c in self.ProbClasses:
            self.ProbClasses[c] /= len(labeled_trainingset)

    def calculateFeatureProbabilities(self, labeled_trainingset):
        for c in self.Features:
            features = {}
            for feature in self.Features[c]:
                if feature not in features:
                    features[feature] = 1
                else:
                    features[feature] += 1
            self.Features[c] = features

        for c in self.Features:
            features = {}
            for feature in self.Features[c]:
                features[feature] = (
                    (self.Features[c][feature] + 1) / (sum(self.Features[c].values()) + self.vocabSize))
            self.ProbFeatures[c] = features

    def classify(self, featureSet, Verbose=True):
        "featureSet: Array aus den Features"
        probabilities = []
        for c in self.ProbClasses:
            prob = 1
            for feature in featureSet:
                if feature not in self.ProbFeatures[c]:
                    prob += log((1 / (sum(self.Features[c].values()) + self.vocabSize)))
                else:
                    prob += log(self.ProbFeatures[c][feature])

            prob += log(self.ProbClasses.get(c))
            probabilities.append((prob, c))
        prediction = max(probabilities, key=itemgetter(0))
        if Verbose:
            print("Vorhergesagte Gesamtbewertung: " + str(prediction[1]))
            print("logarithmische Wahrscheinlichkeit: " + str(prediction[0]) + "\n")
        return prediction


    def classifyAll(self, testset):
        predictions = []
        for entry in testset:
            predictions.append((self.classify(entry[0], False)[1], entry[1]))
        correct_classifications = list(map(lambda x: x[0] == x[1], predictions)).count(True)
        print("Korrekte Klassifikationen: " + str(correct_classifications) + " von " + str(len(predictions)))
        print("Accuracy: " + str(correct_classifications/len(predictions)) + "\n")

if __name__ == "__main__":
    labeled_data = readData("reviews.csv")
    random.shuffle(labeled_data)
    trainingset, testset = labeled_data[:int((0.8 * len(labeled_data)))], labeled_data[int((0.8 * len(labeled_data))):]

    classifier = NaiveBayesClassifier()
    classifier.train(trainingset)
    classifier.classify(["4","5","4","5"])
    classifier.classifyAll(testset)
