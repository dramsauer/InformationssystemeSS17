bsp = [([4, 5, 4, 5], 4, 5), ([3, 4, 5, 4], 4)]


class NaiveBayesClassifier:
    def train(self, labeled_trainingset):
        "Trainingset:[([4,5,4,5],Klasse),...]"
        ProbClasses = {}
        for set in labeled_trainingset:
