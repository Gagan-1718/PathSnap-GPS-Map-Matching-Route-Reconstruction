@echo off
REM Rebuild everything from the raw data, in order. Run from the repository root.
python -m data_processing.explore_data || exit /b 1
python -m feature_engineering.build_dataset || exit /b 1
python -m model_training.train_models || exit /b 1
python -m model_evaluation.evaluate_models || exit /b 1
python -m hmm_viterbi.tune_decoder random_forest || exit /b 1
python -m hyperparameter_optimisation.bayes_opt random_forest 20 || exit /b 1
python -m hmm_viterbi.evaluate_hmm || exit /b 1
python -m visualisations.plots || exit /b 1
python -m visualisations.folium_maps random_forest_tuned || exit /b 1
