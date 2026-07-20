import pynapple as nap
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix, make_scorer, recall_score, precision_score
from sklearn.inspection import permutation_importance

def prepare_data_for_decoding(spikes, onsets, tmin, tmax):
    """
    Prepare neural data for decoding by creating trial-by-trial response matrices
    
    Parameters:
    -----------
    spikes : pynapple.TsGroup
        Spike times for all neurons
    onsets : dict
        Dictionary with odor onset times for each odor
    tmin : float
        Start time of the response window (in seconds)
    tmax : float
        End time of the response window (in seconds)
    
    Returns:
    --------
    X : np.array
        Matrix of shape (n_trials, n_neurons) containing firing rates
    y : np.array
        Vector of trial labels
    """
    all_responses = []
    all_labels = []
    
    # For each odor
    for odor, onset_times in onsets.items():
        tcenter = onset_times.start + 0.5*(tmin + tmax)
        # This ugly hack is because of the annoying time_support rework
        # TODO: Using compute_perievent_continuous in a smart way to get rid of this hack
        max_end = spikes.time_support.end.tolist()
        max_end.append(np.ceil(tcenter[-1] + tmax))
        max_end = np.max(max_end)
        min_start = spikes.time_support.start.tolist()
        min_start.append(np.floor(tcenter[0] + tmin))
        min_start = np.min(min_start)
        temp_iv = nap.IntervalSet(start=min_start, end=max_end)
        trial_peth = nap.compute_perievent(timestamps = spikes, tref=nap.Ts(t=tcenter, time_units="s", time_support=temp_iv), \
            minmax=0.5*(tmax-tmin), time_unit="s")
        trial_responses = []
        # For each neuron
        for unit_idx in spikes.keys():
            response = (trial_peth[unit_idx].count(tmax-tmin))/(tmax-tmin)
            trial_responses.append(response.values.flatten())
            
        # Stack responses for all neurons
        trial_responses = np.array(trial_responses).T  # shape: (n_trials, n_neurons)
        all_responses.append(trial_responses)
        all_labels.extend([odor] * len(onset_times))
    
    X = np.vstack(all_responses)
    y = np.array(all_labels)
    
    return X, y

    # # Apply 0-1 normalization per neuron
    # X_min = X.min(axis=0)
    # X_max = X.max(axis=0)
    # X_range = X_max - X_min
    # # Avoid division by zero
    # X_range[X_range == 0] = 1
    # X_normalized = (X - X_min) / X_range
    
    # return X_normalized, y

def get_classifier(classifier_type='nb', **kwargs):
    """
    Get the specified classifier with optional parameters
    
    Parameters:
    -----------
    classifier_type : str
        Type of classifier ('nb' for Naive Bayes, 'lda' for LDA, 
        'knn' for K-Nearest Neighbors, 'rf' for Random Forest)
    kwargs : dict
        Additional parameters for the classifier
        For RF, common parameters include:
        - n_estimators: number of trees (default 100)
        - max_depth: maximum depth of trees
        - min_samples_split: minimum samples required to split
        - min_samples_leaf: minimum samples required at a leaf node
    """
    if classifier_type.lower() == 'nb':
        return GaussianNB(**kwargs)
    elif classifier_type.lower() == 'lda':
        return LinearDiscriminantAnalysis(**kwargs)
    elif classifier_type.lower() == 'knn':
        return KNeighborsClassifier(**kwargs)
    elif classifier_type.lower() == 'rf':
        return RandomForestClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

def cross_validate_decoder_with_tuning(X, y, classifier_type='rf', n_outer_folds=5, n_inner_folds=4, 
                                  param_grid=None, imbalance_threshold=0.1, random_state=42):
    """
    Perform nested cross-validation with handling for class imbalance using F1 score
    
    Parameters:
    -----------
    X : np.array
        Matrix of shape (n_trials, n_neurons) containing firing rates
    y : np.array
        Vector of trial labels
    classifier_type : str
        Type of classifier ('rf' for Random Forest, etc.)
    n_outer_folds : int
        Number of folds for outer CV (evaluation)
    n_inner_folds : int
        Number of folds for inner CV (model selection)
    param_grid : dict
        Dictionary of parameters to search
    imbalance_threshold : float
        Maximum allowed difference in class proportions (0.1 = 10%)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict containing all results including:
    - Test metrics for each outer fold (accuracy and F1 score)
    - Best parameters for each outer fold
    - Feature importances for each outer fold
    - Class distribution information
    """
    
    # Check class distribution
    unique_labels, label_counts = np.unique(y, return_counts=True)
    class_proportions = label_counts / len(y)
    max_prop = np.max(class_proportions)
    min_prop = np.min(class_proportions)
    
    # Determine if there's significant imbalance
    is_imbalanced = (max_prop - min_prop) > imbalance_threshold
    
    # Choose appropriate scoring metric based on class balance
    if is_imbalanced:
        # Use macro-averaged F1 score for imbalanced data to give equal importance to all classes
        # Explicitly use average='macro' without pos_label for multi-class problems
        def macro_f1(y_true, y_pred):
            return f1_score(y_true, y_pred, average='macro')
        
        scoring_metric = make_scorer(macro_f1)
        scoring_name = 'f1_macro'
    else:
        # Use accuracy for balanced data
        scoring_metric = 'accuracy'
        scoring_name = 'accuracy'
    
    # Default parameter grids for different classifiers
    if param_grid is None:
        if classifier_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200, 400],
                'max_depth': [None, 10, 20, 40],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'class_weight': ['balanced'] if is_imbalanced else [None]
            }
        elif classifier_type == 'knn':
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        elif classifier_type == 'lda':
            param_grid = {
                'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': [None, 'auto']
            }
    
    # Initialize cross-validation strategy based on class balance
    # Always use stratified CV for both balanced and imbalanced to maintain class proportions
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
    
    # Storage for results
    outer_accuracies = []
    outer_f1_scores = []
    best_params_list = []
    feature_importance_list = []
    conf_mats = []
    fold_predictions = {}
    
    # Outer loop
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]
        
        # Initialize classifier for grid search
        if classifier_type == 'rf':
            clf = RandomForestClassifier(random_state=random_state)
        elif classifier_type == 'knn':
            clf = KNeighborsClassifier()
        elif classifier_type == 'lda':
            clf = LinearDiscriminantAnalysis()
        elif classifier_type == 'nb':
            clf = GaussianNB()
        
        # Perform grid search on training data if param_grid is provided
        if param_grid is not None:
            grid_search = GridSearchCV(
                clf, param_grid,
                cv=inner_cv,
                scoring=scoring_metric,  # Use F1 for imbalanced data, accuracy otherwise
                n_jobs=-1
            )
            grid_search.fit(X_train_outer, y_train_outer)
            
            # Get best parameters and refit model
            best_params = grid_search.best_params_
            best_params_list.append(best_params)
            
            # Train best model on full training data
            best_model = grid_search.best_estimator_
        else:
            # If no param_grid, just fit the base model
            best_model = clf
            best_model.fit(X_train_outer, y_train_outer)
            best_params_list.append({})
        
        # Evaluate on test set using both metrics
        y_pred = best_model.predict(X_test_outer)
        if is_imbalanced:
            test_accuracy = balanced_accuracy_score(y_test_outer, y_pred)
        else:
            test_accuracy = accuracy_score(y_test_outer, y_pred)
        
        # Handle case where a class might be missing in the predictions
        try:
            test_f1 = f1_score(y_test_outer, y_pred, average='macro')
        except Exception as e:
            print(f"Warning in fold {fold_idx}: {str(e)}")
            # If there's an error calculating F1, log it but continue with a NaN value
            test_f1 = float('nan')
        
        outer_accuracies.append(test_accuracy)
        outer_f1_scores.append(test_f1)
        
        # Store confusion matrix
        conf_mats.append(confusion_matrix(y_test_outer, y_pred, labels=unique_labels))
        
        # Store predictions
        fold_predictions[fold_idx] = {
            'test_indices': test_idx,
            'y_true': y_test_outer,
            'y_pred': y_pred
        }
        
        # Get feature importance if it's Random Forest
        if classifier_type == 'rf':
            feature_importance_list.append(best_model.feature_importances_)
    
    # Calculate final results
    results = {
        'test_accuracies': outer_accuracies,
        'mean_accuracy': np.mean(outer_accuracies),
        'std_accuracy': np.std(outer_accuracies),
        'test_f1_scores': outer_f1_scores,
        'mean_f1_score': np.mean(outer_f1_scores),
        'std_f1_score': np.std(outer_f1_scores),
        'optimization_metric': scoring_name,
        'best_params_per_fold': best_params_list,
        'confusion_matrices': conf_mats,
        'fold_predictions': fold_predictions,
        'feature_importances': feature_importance_list if classifier_type == 'rf' else None,
        'class_distribution': {
            'labels': unique_labels,
            'counts': label_counts,
            'proportions': class_proportions,
            'is_imbalanced': is_imbalanced,
            'imbalance_measure': max_prop - min_prop
        }
    }
    
    return results

def cross_validate_single_odor_decoder(X, y, target_odor, classifier_type='rf', n_outer_folds=5, n_inner_folds=4, 
                               param_grid=None, random_state=42):
    """
    Perform nested cross-validation for binary classification to decode one target odor vs all others.
    
    Parameters:
    -----------
    X : np.array
        Matrix of shape (n_trials, n_neurons) containing firing rates
    y : np.array
        Vector of trial labels
    target_odor : str or int
        The label of the target odor to decode (e.g., 'Odor D')
    classifier_type : str
        Type of classifier ('rf' for Random Forest, etc.)
    n_outer_folds : int
        Number of folds for outer CV (evaluation)
    n_inner_folds : int
        Number of folds for inner CV (model selection)
    param_grid : dict
        Dictionary of parameters to search
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    dict containing all results including:
    - Test metrics for each outer fold
    - Best parameters for each outer fold
    - Feature importances for each outer fold
    - Class distribution information
    """
    # Check if target odor exists in the dataset
    unique_labels = np.unique(y)
    if target_odor not in unique_labels:
        raise ValueError(f"Target odor '{target_odor}' not found in the dataset. Available classes: {unique_labels}")
    
    # Create binary labels (target_odor vs. others)
    y_binary = np.array([1 if label == target_odor else 0 for label in y])
    
    # Print class distribution
    target_count = np.sum(y_binary == 1)
    other_count = np.sum(y_binary == 0)
    print(f"Binary classification: '{target_odor}' ({target_count} trials) vs. Others ({other_count} trials)")
    print(f"Target odor proportion: {target_count / len(y_binary):.2f}")
    
    # Create a custom scorer that balances recall and precision for the target odor
    def target_f1(y_true, y_pred):
        # Calculate F1 specifically for the positive class (target odor)
        return f1_score(y_true, y_pred, pos_label=1, average='binary')
    
    def target_recall(y_true, y_pred):
        # Calculate recall specifically for the positive class (target odor)
        return recall_score(y_true, y_pred, pos_label=1, average='binary')
        
    # Choose scoring metric that focuses on target odor detection
    scoring_metric = make_scorer(target_f1)
    scoring_name = f'f1_{target_odor}'
    
    # Default parameter grids for different classifiers
    if param_grid is None:
        if classifier_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]  # Try different weights for target class
            }
        elif classifier_type == 'knn':
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        elif classifier_type == 'lda':
            param_grid = {
                'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': [None, 'auto']
            }
    
    # Initialize cross-validation strategy
    # Always use stratified CV to maintain class proportions in the binary problem
    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
    
    # Storage for results
    outer_balanced_accuracies = []
    outer_f1_scores = []
    outer_recalls = []
    outer_precisions = []
    best_params_list = []
    feature_importance_list = []
    conf_mats = []
    fold_predictions = {}
    
    # Outer loop
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_binary)):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y_binary[train_idx], y_binary[test_idx]
        
        # Initialize classifier for grid search
        if classifier_type == 'rf':
            clf = RandomForestClassifier(random_state=random_state)
        elif classifier_type == 'knn':
            clf = KNeighborsClassifier()
        elif classifier_type == 'lda':
            clf = LinearDiscriminantAnalysis()
        elif classifier_type == 'nb':
            clf = GaussianNB()
        
        # Perform grid search on training data if param_grid is provided
        if param_grid is not None:
            grid_search = GridSearchCV(
                clf, param_grid,
                cv=inner_cv,
                scoring=scoring_metric,
                n_jobs=-1
            )
            grid_search.fit(X_train_outer, y_train_outer)
            
            # Get best parameters and refit model
            best_params = grid_search.best_params_
            best_params_list.append(best_params)
            
            # Train best model on full training data
            best_model = grid_search.best_estimator_
        else:
            # If no param_grid, just fit the base model
            best_model = clf
            best_model.fit(X_train_outer, y_train_outer)
            best_params_list.append({})
        
        # Evaluate on test set using multiple metrics
        y_pred = best_model.predict(X_test_outer)
        
        # Calculate metrics
        test_balanced_accuracy = balanced_accuracy_score(y_test_outer, y_pred)
        
        # Try/except blocks to handle potential edge cases
        try:
            # Set zero_division=0 to handle cases where no positive samples are predicted
            test_f1 = f1_score(y_test_outer, y_pred, pos_label=1, average='binary', zero_division=0)
        except Exception as e:
            print(f"Warning in fold {fold_idx}: {str(e)}")
            test_f1 = 0.0
            
        try:
            # Set zero_division=0 to handle cases where no positive samples are predicted
            test_recall = recall_score(y_test_outer, y_pred, pos_label=1, average='binary', zero_division=0)
        except Exception as e:
            print(f"Warning in fold {fold_idx}: {str(e)}")
            test_recall = 0.0
            
        try:
            # Set zero_division=0 to handle cases where no positive samples are predicted
            test_precision = precision_score(y_test_outer, y_pred, pos_label=1, average='binary', zero_division=0)
        except Exception as e:
            print(f"Warning in fold {fold_idx}: {str(e)}")
            test_precision = 0.0
        
        # Store metrics
        outer_balanced_accuracies.append(test_balanced_accuracy)
        outer_f1_scores.append(test_f1)
        outer_recalls.append(test_recall)
        outer_precisions.append(test_precision)
        
        # Store confusion matrix
        conf_mats.append(confusion_matrix(y_test_outer, y_pred, labels=[0, 1]))
        
        # Store predictions
        fold_predictions[fold_idx] = {
            'test_indices': test_idx,
            'y_true': y_test_outer,
            'y_pred': y_pred,
            'y_true_original': [y[i] for i in test_idx]  # Store original odor labels
        }
        
        # Get feature importance if it's Random Forest
        if classifier_type == 'rf':
            feature_importance_list.append(best_model.feature_importances_)
    
    # Calculate final results
    results = {
        'test_balanced_accuracies': outer_balanced_accuracies,
        'mean_balanced_accuracy': np.nanmean(outer_balanced_accuracies),
        'std_balanced_accuracy': np.nanstd(outer_balanced_accuracies),
        'test_f1_scores': outer_f1_scores,
        'mean_f1_score': np.nanmean(outer_f1_scores),
        'std_f1_score': np.nanstd(outer_f1_scores),
        'test_recalls': outer_recalls,
        'mean_recall': np.nanmean(outer_recalls),
        'std_recall': np.nanstd(outer_recalls),
        'test_precisions': outer_precisions,
        'mean_precision': np.nanmean(outer_precisions),
        'std_precision': np.nanstd(outer_precisions),
        'optimization_metric': scoring_name,
        'best_params_per_fold': best_params_list,
        'confusion_matrices': conf_mats,
        'fold_predictions': fold_predictions,
        'feature_importances': feature_importance_list if classifier_type == 'rf' else None,
        'target_odor': target_odor,
        'class_distribution': {
            'target_count': target_count,
            'other_count': other_count,
            'target_proportion': target_count / len(y_binary)
        }
    }
    
    return results

def plot_nested_cv_results(results, classifier_name="", block_name=""):
    """
    Plot results from nested cross-validation
    """
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy distribution
    plt.subplot(131)
    plt.boxplot(results['test_scores'])
    plt.title(f'{block_name} {classifier_name}\nNested CV Accuracy Distribution')
    plt.ylabel('Accuracy')
    plt.xticks([1], ['Test Scores'])
    
    # Plot average confusion matrix
    plt.subplot(132)
    avg_conf_mat = np.mean(results['confusion_matrices'], axis=0)
    normalized_conf_mat = avg_conf_mat / np.sum(avg_conf_mat, axis=1)[:, np.newaxis]
    sns.heatmap(normalized_conf_mat,
                annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Average Confusion Matrix\nMean Accuracy: {results["mean_score"]:.2f} ± {results["std_score"]:.2f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Plot feature importance if available
    if results['feature_importances'] is not None:
        plt.subplot(133)
        mean_importance = np.mean(results['feature_importances'], axis=0)
        std_importance = np.std(results['feature_importances'], axis=0)
        plt.bar(range(len(mean_importance)), mean_importance,
                yerr=std_importance, capsize=5)
        plt.title('Feature Importance Across Folds')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
    
    plt.tight_layout()


def compare_classifiers(X, y, odor_labels, classifiers=['nb', 'lda', 'knn', 'rf'], block_name="", **kwargs):
    """
    Compare performance of different classifiers
    
    Parameters:
    -----------
    X : np.array
        Matrix of shape (n_trials, n_neurons) containing firing rates
    y : np.array
        Vector of trial labels
    odor_labels : list
        List of odor labels
    classifiers : list
        List of classifier types to compare
    block_name : str
        Name of the block for plot titles
    kwargs : dict
        Additional parameters for classifiers
    """
    results = {}
    classifier_names = {'nb': 'Naive Bayes', 'lda': 'LDA', 'knn': 'KNN', 'rf': 'Random Forest'}
    
    for clf_type in classifiers:
        # Run cross-validation
        cv_results = cross_validate_decoder(X, y, classifier_type=clf_type, **kwargs)
        
        # Plot results
        norm_conf_mat, std_conf_mat = plot_cv_results(
            cv_results['accuracies'], 
            cv_results['conf_mats'], 
            cv_results['y_true_all'], 
            cv_results['y_pred_all'], 
            odor_labels, block_name,
            classifier_names[clf_type]
        )
        
        # Store all results
        results[clf_type] = {
            'cv_results': cv_results,
            'norm_conf_mat': norm_conf_mat,
            'std_conf_mat': std_conf_mat
        }
        
        print(f"{block_name} {classifier_names[clf_type]} CV accuracy: "
              f"{np.mean(cv_results['accuracies']):.2f} ± "
              f"{np.std(cv_results['accuracies']):.2f}")
    
    return results

def get_rf_feature_importance(X, y, rf_params=None):
    """
    Calculate and plot feature importance using both built-in importance
    and permutation importance from Random Forest
    
    Parameters:
    -----------
    X : np.array
        Matrix of shape (n_trials, n_neurons) containing firing rates
    y : np.array
        Vector of trial labels
    rf_params : dict
        Parameters for Random Forest classifier
        
    Returns:
    --------
    dict containing feature importance metrics
    """
    if rf_params is None:
        rf_params = {'n_estimators': 100}
    
    # Train Random Forest
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X, y)
    
    # Get built-in feature importance
    builtin_importance = rf.feature_importances_
    
    # Calculate permutation importance
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    perm_importance = result.importances_mean
    
    # Plot both importance metrics
    plt.figure(figsize=(12, 5))
    
    # Built-in importance
    plt.subplot(121)
    plt.bar(range(len(builtin_importance)), builtin_importance)
    plt.title('Built-in Feature Importance')
    plt.xlabel('Neuron Index')
    plt.ylabel('Importance')
    
    # Permutation importance
    plt.subplot(122)
    plt.bar(range(len(perm_importance)), perm_importance)
    plt.title('Permutation Importance')
    plt.xlabel('Neuron Index')
    plt.ylabel('Importance')
    
    plt.tight_layout()
    
    return {
        'builtin_importance': builtin_importance,
        'permutation_importance': perm_importance,
        'perm_importance_std': result.importances_std
    }

def tune_rf_parameters(X, y, param_grid=None):
    """
    Perform grid search to find optimal Random Forest parameters
    
    Parameters:
    -----------
    X : np.array
        Matrix of shape (n_trials, n_neurons) containing firing rates
    y : np.array
        Vector of trial labels
    param_grid : dict
        Dictionary of parameters to search
        
    Returns:
    --------
    dict containing best parameters and CV results
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    # Create Random Forest classifier
    rf = RandomForestClassifier()
    
    # Perform grid search
    grid_search = GridSearchCV(
        rf, param_grid, cv=5,
        scoring='accuracy',
        n_jobs=-1  # Use all available cores
    )
    
    grid_search.fit(X, y)
    
    # Plot parameter importance
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Create plots for each parameter
    n_params = len(param_grid)
    plt.figure(figsize=(15, 5))
    
    for i, param in enumerate(param_grid.keys(), 1):
        plt.subplot(1, n_params, i)
        param_values = [eval(params)[param] 
                       for params in results['params']]
        
        # Calculate mean score for each parameter value
        param_scores = {}
        for value, score in zip(param_values, results['mean_test_score']):
            if value not in param_scores:
                param_scores[value] = []
            param_scores[value].append(score)
        
        values = list(param_scores.keys())
        scores = [np.mean(param_scores[v]) for v in values]
        
        plt.plot(range(len(values)), scores, 'o-')
        plt.xlabel(param)
        plt.ylabel('Mean CV Score')
        plt.xticks(range(len(values)), values, rotation=45)
    
    plt.tight_layout()
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': results
    }

def analyze_rf_predictions(model, X, y, feature_names=None):
    """
    Analyze Random Forest predictions and feature contributions
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained Random Forest model
    X : np.array
        Input features
    y : np.array
        True labels
    feature_names : list
        Names of features (optional)
    """
    # Get predictions from all trees
    predictions = np.array([tree.predict(X) for tree in model.estimators_])
    
    # Calculate prediction confidence
    pred_proba = model.predict_proba(X)
    
    # Plot prediction confidence distribution
    plt.figure(figsize=(15, 5))
    
    # Confidence distribution
    plt.subplot(131)
    plt.hist(np.max(pred_proba, axis=1), bins=20)
    plt.xlabel('Maximum Prediction Probability')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    
    # Tree agreement
    plt.subplot(132)
    agreement = np.mean([predictions[i] == model.predict(X) 
                        for i in range(len(model.estimators_))], axis=0)
    plt.hist(agreement, bins=20)
    plt.xlabel('Fraction of Trees in Agreement')
    plt.ylabel('Count')
    plt.title('Tree Agreement Distribution')
    
    # Feature importance correlation
    plt.subplot(133)
    importances = []
    for tree in model.estimators_:
        importances.append(tree.feature_importances_)
    importances = np.array(importances)
    
    # Calculate correlation between feature importances
    importance_corr = np.corrcoef(importances.T)
    sns.heatmap(importance_corr, cmap='coolwarm', center=0)
    plt.title('Feature Importance Correlation')
    
    plt.tight_layout()
    
    return {
        'prediction_confidence': pred_proba,
        'tree_agreement': agreement,
        'importance_correlation': importance_corr
    }
