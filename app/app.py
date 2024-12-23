import os
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# Настройки
UPLOAD_FOLDER = 'app/static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Инициализация Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Функция проверки расширения файла
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Главная страница
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Проверяем, загрузил ли пользователь файл
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Сохраняем файл
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            results = model([input_path])  # return a list of Results objects
            print(input_path)

            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'negative_' + filename)
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
                result.show()  # display to screen
                result.save(filename=f"{output_path}")  # save to disk

            # Преобразуем изображение в негатив
            # output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'negative_' + filename)
            # with Image.open(input_path) as img:
            #     negative_img = ImageOps.invert(img.convert('RGB'))
            #     negative_img.save(output_path)

            return render_template('index.html', original_image=filename, processed_image='negative_' + filename)

    return render_template('index.html', original_image=None, processed_image=None)

# Отдаём файл для скачивания
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# Запуск приложения
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
