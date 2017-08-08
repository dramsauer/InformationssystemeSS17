bsp = [([4, 5, 4, 5], 4), ([3, 4, 5, 4], 5)]


class NaiveBayesClassifier:
    def train(self, labeled_trainingset):
        "Trainingset:[([4,5,4,5],Klasse),...]"
        ProbClasses = {}
        for s in labeled_trainingset:
            if s[1] not in ProbClasses:
                ProbClasses[s[1]] = 1
            else:
                ProbClasses[s[1]]+=1
        for c in ProbClasses:
            ProbClasses[c] /= len(labeled_trainingset)
        print(ProbClasses)

classifier = NaiveBayesClassifier()
classifier.train(bsp)
