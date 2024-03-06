"""
Convolutional models to be optimized for NCS2.
"""

# import dependencies
from ast import literal_eval
import tensorflow as tf

def get_model(*args, **kwargs):
    if args[2] == 'vgg_mini':
        return vgg_mini(*args, **kwargs)
    elif args[2] == 'vgg16':
        return vgg16(*args, **kwargs)
    elif args[2] == 'vgg19':
        return vgg19(*args, **kwargs)
    elif args[2] == 'mobilenet':
        return mobilnet(*args, **kwargs)
    elif args[2] == 'vgg_med':
        return vgg_med(*args, **kwargs)
    else:
        raise ValueError('Model name not found.')

def vgg_mini(*args, **kwargs):
    print("Building VGG-mini model...")
    print(f'args ---> {args}')
    print(f'kwargs ---> {kwargs}')

    stride_len = literal_eval(kwargs.get('strides', '(2,2)'))
    pooling = literal_eval(kwargs.get('pooling', '(2,2)'))
    kernel = literal_eval(kwargs.get('kernel', '(3,3)'))
    model_name = kwargs.get('pretrain', None)

    model = tf.keras.models.Sequential([

        tf.keras.layers.Input(shape=args[0]),

        tf.keras.layers.Conv2D(
            filters=8, kernel_size=kernel, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=pooling, strides=stride_len),

        tf.keras.layers.Conv2D(
            filters=16, kernel_size=kernel, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=pooling, strides=stride_len),

        tf.keras.layers.Conv2D(
            filters=24, kernel_size=kernel, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=pooling, strides=stride_len),

        tf.keras.layers.Conv2D(
            filters=32, kernel_size=kernel, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=pooling, strides=stride_len),

        tf.keras.layers.Conv2D(
            filters=48, kernel_size=kernel, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=pooling, strides=stride_len),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32,activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=16,activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=args[1], activation="softmax")

    ])

    pretrain = literal_eval(kwargs.get('load_weights', 'False'))
    if  pretrain == True:
        model_path = f'./out/{model_name}/model.h5'
        print(f"Loading weights and activations from {model_path}...")
        # create the base model from pre-trained vgg_mini
        base_model = tf.keras.models.load_model(model_path)
        base_model.trainable = True
        #base_model.summary()

        # Freeze the first N layers specified in config
        N = 8
        for i in range(0, N):
            print(f"  [{i}] Copying layer weights from {base_model.layers[i].name} -----> {model.layers[i].name}")
            model.layers[i].set_weights(base_model.layers[i].get_weights())
            model.layers[i].trainable = False
            base_model.layers[i].trainable = False
    else:
        print("A new model will be trained..")

    ### # display the model summary
    model.summary()

    return model

def vgg_med(*args, **kwargs):
    print("Building VGG-mini model...")
    print(f'args ---> {args}')
    print(f'kwargs ---> {kwargs}')

    stride_len = literal_eval(kwargs.get('strides', '(1,1)'))
    pooling = literal_eval(kwargs.get('pooling', '(2,2)'))
    kernel = literal_eval(kwargs.get('kernel', '(3,3)'))
    model_name = kwargs.get('pretrain', None)

    model = tf.keras.models.Sequential([

        tf.keras.layers.Input(shape=args[0]),

        tf.keras.layers.Conv2D(
            filters=12, kernel_size=kernel, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=pooling, strides=stride_len),

        tf.keras.layers.Conv2D(
            filters=24, kernel_size=kernel, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=pooling, strides=stride_len),

        tf.keras.layers.Conv2D(
            filters=32, kernel_size=kernel, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=pooling, strides=stride_len),

        tf.keras.layers.Conv2D(
            filters=32, kernel_size=kernel, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=pooling, strides=stride_len),

        tf.keras.layers.Conv2D(
            filters=64, kernel_size=kernel, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=pooling, strides=stride_len),

        tf.keras.layers.Conv2D(
            filters=64, kernel_size=kernel, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=pooling, strides=stride_len),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32,activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=16,activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=args[1], activation="softmax")

    ])

    pretrain = literal_eval(kwargs.get('load_weights', 'False'))
    if  pretrain == True:
        model_path = f'./out/{model_name}/model.h5'
        print(f"Loading weights and activations from {model_path}...")
        # create the base model from pre-trained vgg_mini
        base_model = tf.keras.models.load_model(model_path)
        base_model.trainable = True
        #base_model.summary()

        # Freeze the first N layers specified in config
        N = 9
        for i in range(0, N):
            print(f"  [{i}] Copying layer weights from {base_model.layers[i].name} -----> {model.layers[i].name}")
            model.layers[i].set_weights(base_model.layers[i].get_weights())
            model.layers[i].trainable = False
            base_model.layers[i].trainable = False
    else:
        print("A new model will be trained..")

    ### # display the model summary
    model.summary()

    return model


def vgg16(*args, **kwargs):
    print('\nBuilding VGG16 model...')
    print(f'args ---> {args}')
    print(f'kwargs ---> {kwargs}')

    drop_1 = literal_eval(kwargs.get('dropout_1', '0.0'))
    drop_2 = literal_eval(kwargs.get('dropout_2', '0.0'))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            input_shape=args[0], filters=64,kernel_size=(3,3),padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4096,activation="relu"),
        tf.keras.layers.Dropout(drop_1),
        tf.keras.layers.Dense(units=4096,activation="relu"),
        tf.keras.layers.Dropout(drop_2),
        tf.keras.layers.Dense(units=args[1], activation="softmax")
    ])

    pretrain = literal_eval(kwargs.get('load_weights', 'False'))
    if  pretrain == True:
        print("Loading weights and activations from pre-trained model...")
        # create the base model from pre-trained VGG-16
        base_model = tf.keras.applications.vgg16.VGG16(
            input_shape=args[0], include_top=False, weights='imagenet')
        base_model.trainable = True
        #base_model.summary()

        # Freeze the first 15 layers specified in config
        for i in range(1, 19):
            print(f"  [{i-1}] Copying layer weights from {base_model.layers[i].name} -----> {model.layers[i-1].name}")
            ## try:
            ##     if np.shape(base_model.layers[i].get_weights()[0])[2] == 3:
            ##         model.layers[i+1].set_weights(
            ##             [np.expand_dims(np.mean(base_model.layers[i].get_weights()[0], axis=2), axis=2),
            ##                 base_model.layers[i].get_weights()[1]])
            ##     else:
            ##         model.layers[i+1].set_weights(base_model.layers[i].get_weights())
            ## except:
            model.layers[i-1].set_weights(base_model.layers[i].get_weights())
            model.layers[i-1].trainable = False
            base_model.layers[i].trainable = False
    else:
        print("A new model will be trained..")

    ### # display the model summary
    model.summary()

    return model

def vgg19():
    raise NotImplementedError()

def mobilnet():
    raise NotImplementedError()