import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Проверяем доступность GPU (CUDA или Apple Silicon MPS)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Используемое устройство: {device}")

# 1. Функции для загрузки и отображения изображений
def load_image(img_path, max_size=512, shape=None):
    """Загружает картинку и преобразует её в тензор"""
    image = Image.open(img_path).convert('RGB')
    
    # Изменяем размер, если картинка слишком большая
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
        
    if shape is not None:
        size = shape
        
    # Трансформации: ресайз, перевод в тензор и нормализация для VGG
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])

    # Добавляем размерность батча (batch size) и отправляем на устройство
    image = in_transform(image)[:3,:,:].unsqueeze(0).to(device)
    return image

def im_convert(tensor):
    """Конвертирует тензор обратно в картинку для отображения/сохранения"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    
    # Денормализация
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

import numpy as np # нужен для im_convert

# 2. Загрузка модели VGG19
# Мы берем только часть .features, так как классификатор нам не нужен
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

# Замораживаем веса модели (мы будем обучать только саму картинку)
for param in vgg.parameters():
    param.requires_grad_(False)
    
vgg.to(device)

# 3. Функции для извлечения признаков и матрицы Грама
def get_features(image, model, layers=None):
    """Пропускает картинку через VGG и сохраняет выходы нужных слоев"""
    if layers is None:
        # Слои, которые обычно используются для извлечения стиля и контента
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2', # Слой контента
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    """Вычисляет матрицу Грама (описывает стиль)"""
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# --- ОСНОВНОЙ БЛОК НАСТРОЕК ---

# Пути к вашим файлам
content_path = 'content-512.jpg'
style_path = 'candy-style.jpg'

# Загружаем картинки
content = load_image(content_path).to(device)
# Загружаем стиль, подгоняя его под размер контента (опционально, но помогает)
style = load_image(style_path, shape=content.shape[-2:]).to(device)

# Извлекаем признаки из оригинальных картинок
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Считаем матрицы Грама для каждого слоя стиля
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Создаем "целевую" картинку. Начнем с копии контентной (это ускорит процесс)
target = content.clone().requires_grad_(True).to(device)

# Веса для слоев стиля (позволяют настроить, какие детали стиля важнее)
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

# Баланс между сохранением контента и применением стиля
content_weight = 1  # альфа
style_weight = 1e6  # бета

# Оптимизатор (Адам отлично справляется)
optimizer = optim.Adam([target], lr=0.003)

steps = 2000  # Количество шагов обучения
show_every = 500 # Как часто выводить промежуточный результат

print("Начинаем перенос стиля...")

for ii in range(1, steps + 1):
    
    # Получаем признаки целевой картинки
    target_features = get_features(target, vgg)
    
    # Считаем потери контента
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    # Считаем потери стиля
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        
        # Среднеквадратичная ошибка между матрицами Грама
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (d * h * w)
        
    # Общая ошибка
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # Шаг оптимизатора
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Вывод прогресса
    if ii % show_every == 0:
        print(f"Шаг [{ii}/{steps}] | Total Loss: {total_loss.item():.4f}")

print("Готово!")

# Сохранение и отображение результата
result_image = im_convert(target)

# Сохраняем на диск
plt.imsave('output_styled.jpg', result_image)

# Показываем финальный результат в окне (если запускаете в Jupyter/IDE)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(im_convert(content))
ax1.set_title("Контент (Панда)")
ax1.axis('off')
ax2.imshow(im_convert(style))
ax2.set_title("Стиль (Candy)")
ax2.axis('off')
ax3.imshow(result_image)
ax3.set_title("Результат")
ax3.axis('off')
plt.show()
