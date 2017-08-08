bsp = [([4, 5, 4, 5], 4), ([3, 4, 5, 4], 5), ([2, 3, 4, 5], 4)]


class NaiveBayesClassifier:
    def __init__(self):
        self.ProbClasses = {}
        self.ProbFeatures = {}

    def train(self, labeled_trainingset):
        "Trainingset:[([4,5,4,5],Klasse),...]"
        self.calculateClassProbabilities(labeled_trainingset)
        self.calculateFeatureProbabilities(labeled_trainingset)

    def calculateClassProbabilities(self, labeled_trainingset):
        for s in labeled_trainingset:
            if s[1] not in self.ProbClasses:
                self.ProbClasses[s[1]] = 1
            else:
                self.ProbClasses[s[1]] += 1
        for c in self.ProbClasses:
            self.ProbClasses[c] /= len(labeled_trainingset)

    def calculateFeatureProbabilities(self, labeled_trainingset):
        for c in self.ProbClasses:
            features = {}
            for set in labeled_trainingset:
                if set[1] == c:
                    for data in set[0]:
                        if data not in features:
                            features[data] = 1
                        else:
                            features[data] += 1
            self.ProbFeatures[c] = features
        print(self.ProbFeatures)
        for c in self.ProbFeatures:
            features = {}
            for feature in self.ProbFeatures[c]:
                features[feature] = self.ProbFeatures[c][feature] / sum(self.ProbFeatures[c].values())
                print(c)
            self.ProbFeatures[c] = features
        print(self.ProbFeatures)


classifier = NaiveBayesClassifier()
classifier.train(bsp)
