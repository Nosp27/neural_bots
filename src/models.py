from net import MessageBot


def models_init():
    models = {
        'k': MessageBot(
            'models/k_model/model_weights.h5',
            history_df='models/k_model/lstm_in.txt',
        ),
        'kb': MessageBot(
            'models/kb_model/kate_babe_model_30_RMSprop_full_history.h5',
            history_df='models/kb_model/kate_babe_lstm_in.txt',
        ),
        'ls': MessageBot(
            'models/neural_liska/weights_lisa_63_iterations_lr0005_with_dr.h5',
            history_df='models/neural_liska/liska_lstm_in.txt',
        ),
    }

    for model in models.values():
        model.translate('Привет')

    return models
