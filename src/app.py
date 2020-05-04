import re

from flask import Flask, render_template, request, send_file
import random
from time import time
import datetime
import waitress
import models as model_store

log_dir = 'message_log.log'
models = model_store.models_init()


def create_app():
    app = Flask(__name__)
    app.add_url_rule('/', 'index', index)

    app.add_url_rule('/neural_katie', 'neural_katie', neural_katie)
    app.add_url_rule('/neural_girl', 'neural_girl', neural_girl)
    app.add_url_rule('/neural_liska', 'neural_liska', neural_liska)
    app.add_url_rule('/neural_aline', 'neural_aline', neural_aline)
    app.add_url_rule('/neural_goga', 'neural_goga', neural_goga)

    app.add_url_rule('/answer', 'answer', answer)
    app.add_url_rule('/script.js', 'script', script)
    app.add_url_rule('/logo/<string:logo_name>', 'logo', logo)

    return app


def index():
    return 'Hello World!'


def neural_katie():
    headers = [
        'Напиши сообщение',
        'Проверим, исправно ли работает эта сучка?',
        'Хмм... Что тут у нас? Напиши сообщение!',
        'Не стесняйся, она нейронная',
    ]
    return render_template(
        'index.html',
        person='Бот-Катя',
        header=headers[random.randint(0, len(headers) - 1)],
        logo_path='logo/k_logo',
        model_name='k'
    )


def neural_girl():
    headers = [
        'Это обновленная нейроночка, которая многому научилась',
        'Этот алгоритм скушал больше двух лет переписки с Катей :3',
        'Хмм... Обожаю этого бота)0) А он еще и с апгрейдом!',
        'Не стесняйся, она нейронная) (но иногда почти как настоящая)',
    ]
    return render_template(
        'index.html',
        person='Бот-Катя (апгрейд) )0)))',
        header=headers[random.randint(0, len(headers) - 1)],
        logo_path='logo/kb_logo?n=5&disperse=%s' % time().hex(),
        model_name='kb'
    )


def neural_liska():
    headers = [
        'Нейролисик, встречайте',
        'Пообщайся с нейронной лиской, не заставляй её скучать'
    ]
    return render_template(
        'index.html',
        person='Бот-Лиска',
        header=headers[random.randint(0, len(headers) - 1)],
        logo_path='logo/lisa_logo?disperse=%s' % time().hex(),
        model_name='ls'
    )


def neural_aline():
    headers = [
        'без комментариев.'
    ]
    return render_template(
        'index.html',
        person='Бот-Алина',
        header=headers[random.randint(0, len(headers) - 1)],
        logo_path='logo/al_logo?disperse=%s' % time().hex(),
        model_name='al'
    )


def neural_goga():
    headers = [
        'Этот вообще бешеный...',
        'Так и не объяснил мне что с деньками',
        'У него выходит какой то новый альбом, мож спросить?',
        'Лежать плюс сосать, скайнэт )0)',
        'Узнай, сколько чёрных в сорочанах'
    ]
    return render_template(
        'index.html',
        person='Бот Георгий Першин',
        header=headers[random.randint(0, len(headers) - 1)],
        logo_path='logo/go_logo?disperse=%s' % time().hex(),
        model_name='go'
    )


def answer():
    try:
        message = request.args['message']
        model_name = request.args.get('model_name')
        write_log(message, model_name)
        if model_name is None:
            return dict(
                status='error',
                message='Internal error! Model not specified!!!',
            )
        model = models[model_name]
        try:
            json_data = dict(
                status='fine',
                message=model.translate(message),
            )
        except KeyError as ke:
            json_data = dict(
                status='error',
                message='Не знаю слова %s' % ke.args[0],
            )
        except Exception as e:
            json_data = dict(
                status='error',
                message='Error. %s: %s' % (type(e), e.args[0]),
            )
        return json_data
    except Exception as e:
        return dict(
            status='error',
            message='Internal error! %s: %s' % (type(e), e.args[0]),
        )


def write_log(message, model_name):
    with open(log_dir, 'a+') as f:
        f.write('\t'.join([
            message.replace('\t', ' '),
            request.remote_addr or 'None',
            request.remote_user or 'None',
            model_name,
            str(datetime.datetime.now()),
        ]
        ) + '\n')


def script():
    with open('static/messageWork.js') as f:
        data = f.read()
    return data


def logo(logo_name):
    assert re.match(r'[a-z_]+', logo_name)
    n = request.args.get('n')
    if n is not None:
        n = int(n)
        filenames = ['../static/%s_%s.jpg' % (logo_name, i) for i in range(n)]
        filename = filenames[random.randint(0, len(filenames) - 1)]
    else:
        filename = '../static/%s.jpg' % logo_name
    return send_file(filename)


if __name__ == '__main__':
    waitress.serve(create_app(), port=80)
