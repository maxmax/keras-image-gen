import time
import keras_cv
from tensorflow import keras

keras.mixed_precision.set_global_policy("float32")
model = keras_cv.models.StableDiffusion(jit_compile=True)

images = model.text_to_image(
    "Traditional Japanese woodblock prints, sleeping cat, "
    "Mount Fuji, сherry blossoms, "
    "Ukiyo-e, produced between the 17th and the 20th centuries",
    batch_size=1,
)

keras.preprocessing.image.array_to_img(images[0]).show()
