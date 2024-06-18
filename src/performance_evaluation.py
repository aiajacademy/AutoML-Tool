from sklearn.metrics import classification_report, confusion_matrix

class PerformanceEvaluator:
    def evaluate_model(self, model, features, target):
        predictions = model.predict(features)
        report = classification_report(target, predictions)
        matrix = confusion_matrix(target, predictions)
        return {'report': report, 'confusion_matrix': matrix}

