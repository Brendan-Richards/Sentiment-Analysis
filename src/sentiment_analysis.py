
import multiprocessing
import time
import matplotlib.pyplot as plt
from process_data import *
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline

#ensemble models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

#dimensionality reduction
from sklearn.decomposition import TruncatedSVD

# classification models
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# regression models
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression

#metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#Vectorizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def classify_targets(targets):
    return targets.replace(to_replace=4, value=5).replace(to_replace=2, value=1)

def classify_regressor_predictions(values):
    f = 5.0/3
    #print("f=" + str(f))
    new_values = []
    for val in values:
        if(val>f and val<(2*f)):
            new_values.append(3)
        elif(val<=f):
            new_values.append(1)
        else:
            new_values.append(5)
    return new_values


def run_gridsearches(pipelines, parameter_grids, dtrainx, dtrainy, dtestx, dtesty, train_data_size, run_type):
    t0 = time.time()

    verbose_level = 5
    cross_val_num = 3

    if (run_type == "clas"):
        dtrainy = classify_targets(dtrainy)
        dtesty = classify_targets(dtesty)

    for i in range(len(pipelines)):
        print("performing gridsearch " + str(i))
        gs = GridSearchCV(pipelines[i], parameter_grids[i], cv=cross_val_num, n_jobs=multiprocessing.cpu_count(),
                          verbose=verbose_level)
        t1 = time.time()
        gs.fit(dtrainx[:train_data_size], dtrainy[:train_data_size])
        elapsed = time.time() - t1
        print("gridsearch took " + str(elapsed) + " seconds.")
        print("best parameters were:")
        print(gs.best_params_)
        print("model metrics:")
        if (run_type == "clas"):
            print("accuracy score: " + str(accuracy_score(gs.best_estimator_.predict(dtestx), dtesty)))
            plot_confusion_matrix(gs.best_estimator_, dtestx, dtesty)
            plt.show()
        else:
            print("mean_absolute_error: " + str(mean_absolute_error(gs.best_estimator_.predict(dtestx), dtesty)))
            print("mean_squared_error: " + str(mean_squared_error(gs.best_estimator_.predict(dtestx), dtesty)))
            print("guesses in correct range: " + str(
                accuracy_score(classify_regressor_predictions(gs.best_estimator_.predict(dtestx)), dtesty)))
        print("\n")

    ttot = time.time() - t0

    print("all gridsearches together took " + str(ttot) + " seconds\n")

def main():
    dframe, data_train_xo, data_train_yo, data_train_xu, data_train_yu, data_test_y, data_test_x = process_data()


if __name__ == "__main__":
    main()





