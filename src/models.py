from net import MessageBot


def models_init():
    models = {
        'k': MessageBot(
            'models/k_model/model_weights.h5',
            history_df='~/neural_sources/kate_lstm_in.txt',
        ),
        'kb': MessageBot(
            'models/kb_model/kate_babe_model_30_RMSprop_full_history.h5',
            history_df='~/neural_sources/kate_babe_lstm_in.txt',
        ),
        'ls': MessageBot(
            'models/neural_liska/weights_lisa_50_iterations_lr0005_with_dr.h5',
            history_df='~/neural_sources/liska_lstm_in.txt',
        ),
        'al': MessageBot(
            'models/neural_aline/weights_aline_50_iterations_lr0005_with_dr.h5',
            history_df='~/neural_sources/aline_lstm.txt',
        ),
        'go': MessageBot(
            'models/neural_goga/weights_goga_50_iterations_lr0005_with_dr.h5',
            history_df='~/neural_sources/goga_lstm.txt',
        ),
    }

    for model in models.values():
        model.translate('Привет')

    return models
