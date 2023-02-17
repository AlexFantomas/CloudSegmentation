import streamlit as st

st.title('Loopy Clouds')

st.markdown("Ce projet est à l’origine une compétition Kaggle à l’initiative du Max Planck Institute of Meteorology. Le but de cette compétition est de pouvoir identifier quatre types de formations nuageuses (« gravel », « fish », « flower » et « sugar ») sur des images satellite. \n Ces formations nuageuses jouent un rôle déterminant sur le climat et sont difficiles à comprendre et à implémenter dans des modèles climatiques. En classant ces formations nuageuses, les chercheurs espèrent mieux les comprendre et améliorer les modèles existants.")

from PIL import Image
img = Image.open("img1.png")
st.image(img)

st.header("Importation des librairies")

with st.echo():
  import tensorflow as tf
  import tensorflow.keras.backend as K
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.callbacks import Callback, ModelCheckpoint
  from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, ZeroPadding2D, UpSampling2D, Concatenate, Input
  from tensorflow.keras.models import Model, load_model, Sequential
  from tensorflow.keras.utils import Sequence
  from tensorflow.keras.activations import relu, sigmoid
  from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy

  import h5py
  import albumentations as A
  from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion,CenterCrop
  import matplotlib.pyplot as plt

  import pandas as pd 
  import numpy as np 
  import seaborn as sns
  import matplotlib.patches as ptch
  import os
  import random
  import cv2
  import glob
  import multiprocessing
  from copy import deepcopy
  from sklearn.model_selection import train_test_split

st.header("Preprocessing")

st.subheader("Définition des répertoires de travail")

with st.echo():
  data_path = "/home/laborde/Documents/Formation/Datascientest/00_Projet/understanding_cloud_organization"
  test_imgs_folder = os.path.join(data_path, 'test_images')
  train_imgs_folder = os.path.join(data_path, 'train_images')
  train_annot_folder = os.path.join(data_path, 'train_annotations_1')
  checkpoints_folder = os.path.join(data_path, 'Checkpoints')
  optimizer_weights = os.path.join(checkpoints_folder, "Checkpoints/optimizer_state")

st.header("Préparation du csv")

with st.echo():
  path_train = os.path.join(data_path, 'train.csv')

  df_train = pd.read_csv(path_train)

  df_train['id_image'] = df_train['Image_Label'].apply(lambda row: row.split('_')[0])
  df_train['cloud_type'] = df_train['Image_Label'].apply(lambda row: row.split('_')[1])
  df_train = df_train.set_index('id_image')
  df_train['ImageId'] = df_train['Image_Label'].apply(lambda row: row.split('_')[0])

st.dataframe(df_train.head())

st.subheader("Mise en place des train et test sets")

with st.echo():
  train_imgs, test_imgs = train_test_split(df_train['ImageId'].unique(),
                                         test_size = 0.2)

st.write("Nombre d'images train :", len(train_imgs))
st.write("Nombre d'images test :", len(test_imgs))

st.header("Data Augmentation")

with st.echo():
  albumentations_train = A.Compose([
      A.VerticalFlip(), #Effet miroir vertical
      A.Rotate(limit=20, interpolation = 0), #Rotation 
      A.HorizontalFlip()], p=.3) #Effet miroir horizontal

st.info('''
Nous avons essayé de travailler sur l'augmentation de luminosité (RandomBrightness) et le contraste (RandomBrightnessContrast) mais cela détériorait les résultats.
''')

st.header("Création d'un modèle Vanilla")

st.info('''
Nous sommes partis sur un modèle from scratch à ??? de paramètres avec une activation de type 'sigmoïd' en sortie
''')

with st.echo():
  K.clear_session()
  def vanilla_unet() :
      inputs = Input(shape=[256,416,3])
      # Encoder 

      x = Conv2D(32, (5, 5), padding='same')(inputs)
      x = BatchNormalization()(x)
      x = LeakyReLU()(x)
      x = Conv2D(32, (5, 5), padding='same')(x)
      x = BatchNormalization()(x)
      x = LeakyReLU()(x)
      pool1 = MaxPooling2D((2, 2), strides=(2, 2))(x)

      x = Conv2D(64, (5, 5), padding='same')(pool1)
      x = BatchNormalization()(x)
      x = LeakyReLU()(x)
      x = Conv2D(64, (5, 5), padding='same')(x)
      x = BatchNormalization()(x)
      x = LeakyReLU()(x)
      pool2 = MaxPooling2D((2, 2), strides=(2, 2))(x)

      x = Conv2D(128, (3, 3), padding='same')(pool2)
      x = BatchNormalization()(x)
      x = LeakyReLU()(x)
      x = Conv2D(128, (3, 3), padding='same')(x)
      x = BatchNormalization()(x)
      x = LeakyReLU()(x)
      pool3 = MaxPooling2D((2, 2), strides=(2, 2))(x)

      x = Conv2D(256, (3, 3), padding='same')(pool3)
      x = BatchNormalization()(x)
      x = LeakyReLU()(x)
      x = Conv2D(256, (3, 3), padding='same')(x)
      x = BatchNormalization()(x)
      x = LeakyReLU()(x)
      x = Conv2D(256, (3, 3), padding='same')(x)
      x = BatchNormalization()(x)

      # Decoder
      x = ZeroPadding2D(padding=(1,1))(x)
      x = Conv2D(256, (3, 3))(x)
      x = BatchNormalization()(x)
      up1 = UpSampling2D()(x)
      x = Concatenate()([up1, pool2])

      x = ZeroPadding2D(padding=(1,1))(x)
      x = Conv2D(128, (3, 3))(x)
      x = BatchNormalization()(x)
      up2 = UpSampling2D()(x)
      x = Concatenate()([up2, pool1])

      x = ZeroPadding2D(padding=(1,1))(x)
      x = Conv2D(64, (3, 3))(x)
      x = BatchNormalization()(x)
      x = Conv2D(4, (3,3), padding = 'same')(x)
      x = tf.keras.layers.Activation("sigmoid")(x)
      model = Model(inputs= inputs, outputs = x)
      return model



st.code('''
model = vanilla_unet()
model.summary()
''', language = "python")

model = vanilla_unet()
st.write(model.summary())

st.code('''
#Définition des tailles d'input et d'output
output_height = model.output.get_shape()[1]
output_width = model.output.get_shape()[2]

input_height = model.input.get_shape()[1]
input_width = model.input.get_shape()[2]
''', language = "python")

output_height = model.output.get_shape()[1]
output_width = model.output.get_shape()[2]

input_height = model.input.get_shape()[1]
input_width = model.input.get_shape()[2]

st.header("Data generator")

with st.echo():
  class DataGenenerator(Sequence):
      def __init__(self, images_list=None, folder_imgs=train_imgs_folder, folder_annot=train_annot_folder,
                   batch_size=32, shuffle=True, augmentation=None, is_test = False, output_height = output_height, output_width = output_width,
                   resized_height=input_height, resized_width=input_width, num_channels=3, num_classes = 4):
          # Taille du batch
          self.batch_size = batch_size
          # Variable pour le mélange aléatoire des images
          self.shuffle = shuffle
          # Gestion de la data augment
          self.augmentation = augmentation
          #Retourne la liste des images 
          if images_list is None:
              self.images_list = os.listdir(folder_imgs)
          else:
              self.images_list = deepcopy(images_list)
          # Nom du répertoire des images
          self.folder_imgs = folder_imgs
          # Nom du répertoire des annotations
          self.folder_annot = folder_annot
          # Nb d'itérations pour chaque epoch
          self.len = len(self.images_list) // self.batch_size
          # Hauteur de l'image redimensionnée
          self.resized_height = resized_height
          # Largeur de l'image redimensionnée
          self.resized_width = resized_width
          # Profondeur de la dernière dimension (couleurs). Par défaut = 3 (RGB)
          self.num_channels = num_channels
          # Nb de classes 
          self.num_classes = num_classes
          self.is_test = not 'train' in folder_imgs
          #self.encoding_list = dico_encod
          self.output_height = output_height
          self.output_width = output_width

      # Retourne le nombre d'itérations par epoch
      def __len__(self):
          return self.len
    
      # Tri aléatoire des images
      def on_epoch_start(self):
          if self.shuffle:
              random.shuffle(self.images_list)

      def __getitem__(self, idx):
          current_batch = self.images_list[idx * self.batch_size: (idx + 1) * self.batch_size]
          X = np.empty((self.batch_size, self.resized_height, self.resized_width, self.num_channels))
          y = np.empty((self.batch_size, self.output_height, self.output_width, self.num_classes))

          for i, image_name in enumerate(current_batch):
              path_img = os.path.join(self.folder_imgs, image_name)
              path_annot1 =os.path.join(self.folder_annot, image_name.split('.')[0]+ '_Gravel.png')
              path_annot2 =os.path.join(self.folder_annot, image_name.split('.')[0]+ '_Fish.png')
              path_annot3 =os.path.join(self.folder_annot, image_name.split('.')[0]+ '_Flower.png')
              path_annot4 =os.path.join(self.folder_annot, image_name.split('.')[0]+ '_Sugar.png')
             
              img = cv2.imread(path_img).astype(np.float32)
              mask_gravel = cv2.imread(path_annot1, cv2.IMREAD_GRAYSCALE).astype(np.int32)
              mask_fish =cv2.imread(path_annot2, cv2.IMREAD_GRAYSCALE).astype(np.int32)
              mask_flower = cv2.imread(path_annot3, cv2.IMREAD_GRAYSCALE).astype(np.int32)
              mask_sugar = cv2.imread(path_annot4, cv2.IMREAD_GRAYSCALE).astype(np.int32)        

              if not self.augmentation is None:
                  # Application de la data augmentation sur l'image et le masque
                  augmented = self.augmentation(image=img, masks=[mask_gravel, mask_fish, mask_flower, mask_sugar])
                  img = augmented['image']
                  mask_gravel = augmented['masks'][0] 
                  mask_fish =  augmented['masks'][1]
                  mask_flower = augmented['masks'][2]
                  mask_sugar =  augmented['masks'][3]

              mask_gravel = cv2.resize(mask_gravel ,(self.output_width, self.output_height), interpolation =0).astype(np.int32)
              mask_fish = cv2.resize(mask_fish ,(self.output_width, self.output_height), interpolation =0).astype(np.int32)
              mask_flower= cv2.resize(mask_flower ,(self.output_width, self.output_height), interpolation =0).astype(np.int32)
              mask_sugar = cv2.resize(mask_sugar ,(self.output_width, self.output_height), interpolation =0).astype(np.int32)

              mask_gravel = np.expand_dims(mask_gravel, 2)
              mask_fish = np.expand_dims(mask_fish, 2)
              mask_flower = np.expand_dims(mask_flower, 2)
              mask_sugar = np.expand_dims(mask_sugar, 2)

              X[i, :, :, :] = cv2.resize(img, (self.resized_width, self.resized_height), interpolation = 0)/255.0

              # One Hot encdoding
              if not self.is_test:
                seg_labels = np.concatenate((mask_gravel, mask_fish, mask_flower, mask_sugar), axis = 2)
                y[i, :, :, :] = seg_labels
          return X, y

st.header("Loss and metrics functions")

with st.echo():
  def custom_bce(y_true, y_pred, weights = {0 :1, 1:5}):
    # On calcule la bce pour chaque pixel et chaque label du batch : shape (batch_size, img_height, img_width, nb_classes = 4)
      bce = K.binary_crossentropy(tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32))
      # On va maintenant appliquer un poids plus important à une non-détection de formation qu'à la non-détection du background

      ratio = weights[1]/weights[0]
      bce_weighted = tf.where(y_true==1, bce*ratio, bce)

      # On suppose que se tromper sur 4 classes au lieu de 2 est plus "grave" => On somme les erreurs de classification
      bce_summed_over_classes = K.sum(bce_weighted, axis = -1)
      # Dans un 2ème temps on fait la moyenne par pixel
      mean_bce = tf.reduce_mean(bce_summed_over_classes)
      return mean_bce

  def dice_coef(y_true, y_pred, smooth=0.1):
      intersection = K.sum(y_true * y_pred)
      return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

st.header("Callbacks")

with st.echo():
  K.clear_session()

  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath= "checkpoints",
      save_weights_only=True,
      monitor='val_loss',
      save_best_only=True)

  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 5, verbose = 2, min_delta =0.01)
  optimizer = tf.keras.optimizers.Adam(1e-3)

st.header("Setting up training")

with st.echo():
  data_generator_train = DataGenenerator(train_imgs, augmentation=albumentations_train)
  data_generator_train_eval = DataGenenerator(train_imgs, shuffle=False)
  data_generator_val = DataGenenerator(test_imgs, shuffle=False)

st.header("Chargement du modèle")

with st.echo():
  tf.keras.backend.clear_session()

  model.compile(optimizer = optimizer, loss = custom_bce, metrics = [dice_coef, 
                                                                 tf.keras.metrics.MeanIoU(4), 'acc']) 
  model.load_weights(os.path.join(checkpoints_folder, 'weights', 'weights2.h5'))

st.code('''
history2 = model.fit(data_generator_train, steps_per_epoch=data_generator_train.__len__(),
                    callbacks= [model_checkpoint_callback, reduce_lr],
                    epochs = 40,
                    workers = -1,
                    validation_data = data_generator_val,
                    validation_steps = data_generator_val.__len__())
print("Adam weights: ", getattr(model.optimizer, 'weights'))
print("Adam iterations : ", getattr(model.optimizer, 'iterations'))
''', language = "python")

img = Image.open("fit.png")
st.image(img)

st.code('''
model.optimizer.get_config()
''', language = "python")

st.write(model.optimizer.get_config())

st.code('''
plt.plot(range(1,41), history2.history['val_dice_coef'], label = 'val_dice_coef')
plt.plot(range(1,41), history2.history['dice_coef'], label = 'dice_coef')
plt.legend()
''', language = "python")

plot1_img = Image.open("plot1.png")
st.image(plot1_img)

st.code('''
plt.plot(range(1,41), history2.history['val_loss'], label = 'val_loss')
plt.plot(range(1,41), history2.history['loss'], label = 'loss')
plt.legend()
''', language = "python")

plot2_img = Image.open("plot2.png")
st.image(plot2_img)

with st.echo():
  from skimage.color import label2rgb
  from PIL import Image
  import matplotlib.patches as mpatches

  # Fonction pour afficher les images avec leurs masques prédits
  @st.cache
  def visu_images_4classes_test(folder, model, height_graph = 15, width_graph = 10):

    plt.figure(figsize = (12, 12))

    pop_1 = mpatches.Patch(color='blue', label='Gravel', alpha = 0.2)
    pop_2 = mpatches.Patch(color='red', label='Fish', alpha = 0.2)
    pop_3 = mpatches.Patch(color='yellow', label='Flower', alpha = 0.2)
    pop_4 = mpatches.Patch(color='green', label='Sugar', alpha = 0.2)

    #On sélectionne nb_images dans le folder
    im_rep = random.sample(os.listdir(folder), 1)
    path_img = folder + im_rep[0]
    path_img_gravel = train_annot_folder +'/' + im_rep[0].split('.')[0] + '_Gravel.png'
    path_img_fish = train_annot_folder + '/' + im_rep[0].split('.')[0] + '_Fish.png'
    path_img_flower = train_annot_folder + '/' + im_rep[0].split('.')[0] + '_Flower.png'
    path_img_sugar = train_annot_folder + '/' +im_rep[0].split('.')[0] + '_Sugar.png'
    #On met l'image aux bonnes dimensions pour la prédiction (1, 480, 320, 3)
    img = cv2.imread(path_img).reshape(1400, 2100, 3)
    img = cv2.resize(img, (416, 256))
  
    # Redimensionnement de l'image pour prédiction
    img1 = img.reshape(-1, 256, 416, 3)
    # Redimensionnement de l'image pour comparaison avec le futur masque
    img2 = cv2.resize(img,(208, 128))

    #On obtient un masque en shape (38400, 4)
    mask = model.predict(img1)[0]
    
    #On crée les masques de chaque formation
    masked_gravel = mask[:,:,0].reshape(128,208)
    masked_fish = mask[:,:,1].reshape(128,208)
    masked_flower = mask[:,:,2].reshape(128,208)
    masked_sugar = mask[:,:,3].reshape(128,208)

    plt.subplot(3,4,1)
    sns.heatmap(masked_gravel)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,4,2)
    sns.heatmap(masked_fish)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,4,3)
    sns.heatmap(masked_flower)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,4,4)
    sns.heatmap(masked_sugar)
    plt.xticks([])
    plt.yticks([])
  
    #On crée une image de couleurs correspondant à chaque formation
  
    im_mask_gravel = label2rgb(image = img2,label = masked_gravel, colors = ['blue'], alpha = 0.2, kind = 'overlay', bg_label = 0)
    im_mask_fish = label2rgb(image = img2, label = masked_fish, colors = ['red'], alpha = 0.2, kind = 'overlay', bg_label = 0)
    im_mask_flower = label2rgb(image = img2, label = masked_flower, colors = ['yellow'], alpha = 0.2, kind = 'overlay', bg_label = 0)
    im_mask_sugar = label2rgb(image = img2, label = masked_sugar, colors = ['green'], alpha = 0.2, kind = 'overlay', bg_label = 0)

    plt.subplot(3,4,5)
    plt.imshow(im_mask_gravel)
    plt.subplot(3,4,6)
    plt.imshow(im_mask_fish)
    plt.subplot(3,4,7)
    plt.imshow(im_mask_flower)
    plt.subplot(3,4,8)
    plt.imshow(im_mask_sugar)

    plt.subplot(3, 4, 9)
    annot_gravel = cv2.imread(path_img_gravel, cv2.IMREAD_GRAYSCALE)
    plt.imshow(annot_gravel)
    plt.subplot(3, 4, 10)
    annot_fish = cv2.imread(path_img_fish, cv2.IMREAD_GRAYSCALE)
    plt.imshow(annot_fish)
    plt.subplot(3, 4, 11)
    annot_flower = cv2.imread(path_img_flower, cv2.IMREAD_GRAYSCALE)
    plt.imshow(annot_flower)
    plt.subplot(3, 4, 12)
    annot_sugar = cv2.imread(path_img_sugar, cv2.IMREAD_GRAYSCALE)
    plt.imshow(annot_sugar)

    plt.show();

st.code('''
folder = train_imgs_folder + '/'
visu_images_4classes_test(folder, model, height_graph = 15, width_graph = 10)
''', language = "python")

st.info("La visualisation nous permet de voir que la dernière classe ('sugar') semble obtenir des résultats corrects. En revanche, les autres classes sont très mal définies. La classe correctement décrite varie selon l'entrainement complet du modèle. Le modèle semble donc améliorer la fonction de coût uniquement au travers d'une des classes. Nous allons donc essayer d'optimiser notre fonction de coût pour corriger ce problème.") 

folder = train_imgs_folder + '/'

if st.button("Visualiser une image"):
  st.pyplot(visu_images_4classes_test(folder, model, height_graph = 15, width_graph = 10))
  st.balloons()
st.set_option('deprecation.showPyplotGlobalUse', False)



