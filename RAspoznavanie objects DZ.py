import cv2

# Загрузка каскада Хаара для распознавания глаз
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Открытие видеопотока
cap = cv2.VideoCapture(0)

'''Считывание кадра из видеопотока VideoCapture
 ret- был ли кадр успешно считан
 frame - считывает кадр в виде массива NumPy'''
while True:
    ret, frame = cap.read()
    # проверка, был ли считан кадр
    if not ret:
        break

    # Преобразование кадра в черно-белый формат
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение глаз с помощью каскада Хаара
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(eyes) > 1:
        # Объединение координат и размеров обоих глаз
        x1, y1, w1, h1 = eyes[0]
        x2, y2, w2, h2 = eyes[1]
        x = min(x1, x2)-40
        y = min(y1, y2)
        w = max(x1 + w1, x2 + w2) - x+60
        h = max(y1 + h1, y2 + h2) - y

        # Применение размытия к области глаз сплошным прямоугольником
        blurred_eye = cv2.blur(frame[y:y + h, x:x + w], (30, 30))

        # Отображение размытой области глаз на исходном кадре
        frame[y:y + h, x:x + w] = blurred_eye

    cv2.imshow('Blurred eyes', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break