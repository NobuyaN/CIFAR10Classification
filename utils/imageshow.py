import matplotlib.pyplot as plt

def imshow(img):
  # img_f = (img_i - mean) / std
  # img_i = img_f * std + mean = (img_i / 2) + 0.5
  img = (img * 0.5) + 0.5
  transformed_img = img.numpy().transpose((1, 2, 0))
  plt.imshow(transformed_img)
  plt.axis('off')
  plt.show()