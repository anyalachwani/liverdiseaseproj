from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


df = pd.read_csv('/Users/anyalachwani/Downloads/archive/new_fileactual.csv')
input_data = df.iloc[:, :9].to_numpy(dtype='float64')
output_data = df.iloc[:, 9].to_numpy(dtype='float64')

imputer = SimpleImputer(strategy='mean')
input_data_imputed = imputer.fit_transform(input_data)

# Split the data into training and testing sets
inputs_train, inputs_test, target_train, target_test = train_test_split(
    input_data_imputed, output_data, test_size=0.2, random_state=42
)

@ignore_warnings(category=ConvergenceWarning)
def get_best_model(model, grid):
    imputer = SimpleImputer(strategy='mean')
    inputs_train_imputed = imputer.fit_transform(inputs_train)
    grid_search = GridSearchCV(model, grid, cv=5, verbose=2)
    grid_search.fit(inputs_train_imputed, target_train)
    best_parameters = grid_search.best_params_
    best_accscore = grid_search.best_score_

    print("Best Parameters:", best_parameters)
    print("Best accuracy on cv holdout data:", best_accscore)

    best_model = grid_search.best_estimator_
    best_model.fit(inputs_train_imputed, target_train)

    predictions = best_model.predict(inputs_test)
    accuracy = accuracy_score(target_test, predictions)
    print("Accuracy on test data:", accuracy)

    predictions = best_model.predict(inputs_train)
    accuracy = accuracy_score(target_train, predictions)
    print("Accuracy on train data:", accuracy)

    return best_model

def main():
    model_savefile = Path('randomforestmodel.pkl')


    print('Training model...')
    best_model = get_best_model(RandomForestClassifier(), {
        'n_estimators': [20, 100, 200, 500, 750, 1000],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_depth': [None],
    })
    print('Saving model...')
    with open(model_savefile, 'wb') as f:
        pickle.dump(best_model, f)
    print('Success!')

    rf_importances = best_model.feature_importances_
    print("rf_importance", rf_importances)

    df = pd.read_csv('/Users/anyalachwani/Downloads/archive/new_fileactual.csv', encoding='latin-1')
    feature_names = np.array(df.columns[:-1])
    print("feature names", feature_names)


    idx_importance = np.argsort(rf_importances)[::-1]
    rf_importances_sorted = rf_importances[idx_importance]
    feature_names_sorted = feature_names[idx_importance]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names_sorted)), rf_importances_sorted, tick_label=feature_names_sorted)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xticks(range(len(feature_names_sorted)))
    ax.set_xticklabels(feature_names_sorted, rotation=45)

    # permutation importance
    pm_importances = permutation_importance(best_model, inputs_train, target_train, n_repeats=5, random_state=30)
    pm_importances_mean_sorted = pm_importances.importances_mean[idx_importance]
    pm_importances_std_sorted = pm_importances.importances_std[idx_importance]
    # plot together
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.bar(
        np.arange(len(feature_names_sorted)) - 0.15, rf_importances_sorted,
        color='red', label='RF Importances',
        width=0.3
    )
    ax.bar(
        np.arange(len(feature_names_sorted)) + 0.15, pm_importances_mean_sorted,
        color='blue', label='Permutation Importance',
        width=0.3, yerr=pm_importances_std_sorted
    )

    ax.legend()
    plt.show()

# train on most important feature
    accuracies_test = []
    accuracies_train = []
    features = []
    indices = np.argsort(rf_importances)[::-1]
    X_train1, X_dev1, y_train1, y_dev1 = train_test_split(input_data_imputed, output_data, test_size=0.2, random_state=42)

    for i in range(1,4):
        impfeature_indices = indices[:i]
        print("impfeature_indices")
        print(impfeature_indices)
        most_important_feature_data = input_data_imputed[:, impfeature_indices]
        xtrain, xdev = X_train1[:, impfeature_indices], X_dev1[:, impfeature_indices]

        randomforest_mostimportant = RandomForestClassifier()
        randomforest_mostimportant.fit(xtrain, y_train1)
        predictions_mostimportant = randomforest_mostimportant.predict(xtrain)
        accuracy1 = accuracy_score(y_train1, predictions_mostimportant)
        accuracies_train.append(accuracy1)

        predictions_testdata = randomforest_mostimportant.predict(xdev)
        accuracy2 = accuracy_score(y_dev1, predictions_testdata)
        print("accuracy2")
        print(accuracy2)
        accuracies_test.append(accuracy2)

        features.append(i)
        print("features:")
        print(features)
    # On test
    plt.plot(features, accuracies_test, marker='o')
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy on Test data")
    plt.title(" Number of Features vs Accuracy")
    plt.grid(True)
    plt.ylim((0, 1))
    plt.show()

    # On train
    plt.plot(features, accuracies_train, marker='o')
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy on Train data")
    plt.title(" Number of Features vs Accuracy")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
