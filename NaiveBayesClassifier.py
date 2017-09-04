# encoding: utf-8

from csv import reader
from operator import itemgetter
from math import log
import random
import numpy as np

def readData(file):
    """Funktion zum Einlesen einer CSV-Datei mit Komma als Delimiter.
    Die Daten werden in einem Array aus Tupeln mit dem Format ([Bewertung1, Bewertung2, Bewertung3, Bewertung4], Gesamt-
    bewertung) gespeichert und zurückgegeben."""
    file = open(file, "r")
    data = []
    csvReader = reader(file, delimiter=",")
    csvReader.__next__()
    for row in csvReader:
        data.append((row[0:4], row[4]))
    return data


class NaiveBayesClassifier:
    """ Klasse für den Naive-Bayes Klassifizierer."""

    def __init__(self):
        """Deklarieren der Klassenvariablen für Klassen- und Featurewahrscheinlichkeiten"""

        # Dictionary, in dem jede Klasse und ihre zugehörige Wahrscheinlichkeit gespeichert wird.
        self.ProbClasses = {}

        # Dictionary mit den Klassenvariablen als Keys. Jede Klassenvariable hat als Value wieder ein Dictionary mit den
        # Teilbewertungen als Keys und deren absoluten Wahrscheinlichkeiten als Values.
        self.Features = {}

        # Dictionary mit den Klassenvariablen als Keys. Jede Klassenvariable hat als Value wieder ein Dictionary mit den
        # Teilbewertungen als Keys und deren relativen Wahrscheinlichkeiten(Somit also die Bedingten
        # Wahrscheinlichkeiten) als Values.
        self.ProbFeatures = {}

        # Speichern der Anzahl unterschiedlicher Teilbewertungen, benötigt für smoothing
        self.vocabSize = 0

    def train(self, labeled_trainingset):
        """ Methode zum Trainieren des Klassifizierers.
        Parameter labeled_trainingset: Liste aus Tupeln, Tupeln bestehen aus Liste der Features und der Klasse
        Bsp:[([Feature1, Feature2,..],Klasse1),([Feature1, Feature2,..],Klasse2)...]"""
        self.calculateClassProbabilitiesAndSizeOfVocab(labeled_trainingset)
        self.calculateFeatureProbabilities()

    def calculateClassProbabilitiesAndSizeOfVocab(self, labeled_trainingset):
        """ Methode zur Berechnung der Klassenwahrscheinlichkeiten und Größe des Vokabulars( Anzahl verschiedener
        Teilbewertungen)"""
        vocab = []
        for entry in labeled_trainingset:
            # Für jede Klassenvariable werden alle Teilbewertungen in einem Array in self.Features gespeichert.
            if entry[1] not in self.Features:
                self.Features[entry[1]] = []
            self.Features[entry[1]].extend(entry[0])

            # Die absolute Häufigkeit der Klassen wird gezählt und in self.ProbClasses gespeichert.
            if entry[1] not in self.ProbClasses:
                self.ProbClasses[entry[1]] = 1
            else:
                self.ProbClasses[entry[1]] += 1

            # Alle Teilbewertungen werden in vocab gespeichert.
            vocab.extend(entry[0])

        # Berechnung der Anzahl unterschiedlicher Teilbewertungen.
        self.vocabSize = len(set(vocab))

        # Die relativen Wahrscheinlichkeiten der Klassen werden berechnet und in self.ProbClasses gespeichert.
        for c in self.ProbClasses:
            self.ProbClasses[c] /= len(labeled_trainingset)

    def calculateFeatureProbabilities(self):
        """ Methode zur Berechnung der bedingten Wahrscheinlichkeiten P(Feature|Klasse)"""

        # Die absoluten Häufigkeiten der einzelnen Teilbewertungen in jeder Klasse werden gezählt und in einem
        # Dictionary gespeichert. Anschließend wird dieses Dictionary in self.Features gespeichert
        for c in self.Features:
            features = {}
            for feature in self.Features[c]:
                if feature not in features:
                    features[feature] = 1
                else:
                    features[feature] += 1
            self.Features[c] = features

        # Berechnung der bedingten Wahrscheinlichkeiten für jede Teilbewertung jeder Klasse mithilfe von Laplace-
        # Smoothing und Speicherung in self.ProbFeatures
        for c in self.Features:
            features = {}
            for feature in self.Features[c]:
                # sum(self.Features[c].values() ist die Gesamtanzahl der Teilbewertung einer Klasse.
                features[feature] = (
                    (self.Features[c][feature] + 1) / (sum(self.Features[c].values()) + self.vocabSize + 1))
            self.ProbFeatures[c] = features

    def classify(self, featureSet, Verbose=True):
        """ Methode zur Klassifizierung eines einzelnen Datensatzes.
        Die Formel zur Berechnung der Wahrscheinlichkeit für jede Klasse wird dabei logarithmisch umgeformt, um das
        Rechnen mit sehr kleinen Zahlen zu vermeiden.
        FeatureSet: Array aus den 4 Teilbewertungen eines Restaurants.
        """
        probabilities = []
        for c in self.ProbClasses:
            prob = 0
            # Für jedes Feature wird die Wahrscheinlichkeit ermittelt und aufaddiert(aufgrund der log Umformung)
            for feature in featureSet:
                if feature not in self.ProbFeatures[c]:
                    prob += log((1 / (sum(self.Features[c].values()) + self.vocabSize + 1)))
                else:
                    prob += log(self.ProbFeatures[c][feature])

            # Addition der Klassenwahrscheinlichkeit P(Klasse)
            prob += log(self.ProbClasses.get(c))
            probabilities.append((prob, c))

        # Ermittlung der wahrscheinlichsten Klasse für die übergebenen Daten
        prediction = max(probabilities, key=itemgetter(0))

        if Verbose:
            print("Vorhergesagte Gesamtbewertung: " + str(prediction[1]))
            print("logarithmische Wahrscheinlichkeit: " + str(prediction[0]) + "\n")
        return prediction

    def classifyAll(self, testset, Verbose=False):
        """ Methode zur Klassifizierung eines Datensatzes aus mehreren Restaurants"""
        predictions = []
        # Jedes Restaurant des Datensatzes wird mithilfe der classify-Funktion klassifiziert und die Vorhersage
        # gespeichert
        for entry in testset:
            predictions.append((self.classify(entry[0], False)[1], entry[1]))
        # Berechnung der Anzahl korrekter Klassifizierungen
        correct_classifications = list(map(lambda x: x[0] == x[1], predictions)).count(True)

        # Ausgabe der Anzahl korrekter Klassifizierungen und der Accuracy
        if Verbose:
            print("Korrekte Klassifikationen: " + str(correct_classifications) + " von " + str(len(predictions)))
            print("Accuracy: " + str(correct_classifications / len(predictions)) + "\n")
        return correct_classifications / len(predictions)


    def crossFoldValidation(self, data, k):
        """ Methode zur Kreuzvalidierung des Klassifizieres."""
        random.shuffle(data)
        partitions = []
        # Die Daten werden in k gleich große Mengen aufgeteilt.
        for i in range(0, len(data), k):
            partitions.append(data[i:i + k])
        accuracies = []
        # Nun werden k-Testläufe gestartet, bei denen jeweils die i-te Teilmenge als Testset und die übrigen Teil-
        # mengen als Trainingsset dienen. Für jeden Durchlauf wird die Accuracy gespeichert.
        for i in range(0, k):
            classifier = NaiveBayesClassifier()
            trainingset = []
            for partition in partitions[:i] + partitions[i+1:]:
                trainingset.extend(partition)
            classifier.train(trainingset)
            accuracies.append(classifier.classifyAll(partitions[i]))
        # Ausgabe der Durchschnittlichen Accuracy in den k-Testläufen
        print("Durchschnittliche Accuracy bei Cross Fold Validation: " + str(sum(accuracies) / len(accuracies)))


if __name__ == "__main__":
    # Einlesen der Daten in der csv-Datei
    labeled_data = readData("reviews.csv")

    # Aufteilen der Daten in Trainingsset und Testset im Verhältnis 80:20
    trainingset, testset = labeled_data[:int((0.8 * len(labeled_data)))], labeled_data[int((0.8 * len(labeled_data))):]

    classifier = NaiveBayesClassifier()
    classifier.train(trainingset)
    classifier.classifyAll(testset,True)

    # Kreuzvalidierung
    classifier.crossFoldValidation(labeled_data, 10)



