import os
import tensorflow as tf
root_logdir_ctc=  "../Data/my_logs_ctc"# os.path.join(os.curdir, "../Data/my_logs_ctc")
root_logdir_tr= "../Data/my_logs_tr"#os.path.join(os.curdir, "../Data/my_logs_tr")
def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir_ctc = get_run_logdir(root_logdir= root_logdir_ctc)
tensorboard_cb_ctc = tf.keras.callbacks.TensorBoard(run_logdir_ctc)
run_logdir_tr = get_run_logdir(root_logdir= root_logdir_tr)
tensorboard_cb_tr = tf.keras.callbacks.TensorBoard(run_logdir_tr)

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

earlystopping = tf.keras.callbacks.EarlyStopping(patience=10)
exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                                    monitor='loss',
                                    factor=0.5,
                                    patience=5,
                                    verbose=0,
                                    mode='auto',
                                    min_delta=0.2,
                                    cooldown=0,
                                    min_lr=0)
lr_scheduler_tr = tf.keras.callbacks.ReduceLROnPlateau(
                                    monitor='loss',
                                    factor=0.15,
                                    patience=5,
                                    verbose=0,
                                    mode='auto',
                                    min_delta=0.01,
                                    cooldown=0,
                                    min_lr=0)

#tensorboard --logdir=./my_logs --port=6006
# filepath = '/tmp/checkpoint'
# ckeckpoint = tf.keras.callbacks.ModelCheckpoint(
#     filepath,
#     monitor= "loss",
#     verbose=  0,
#     save_best_only= True,
#     save_weights_only= False,
#     mode="auto",
#     save_freq="epoch",
#     options=None,
#     initial_value_threshold=None,
# )

outputFolder = '../checkpoints'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

filepath=outputFolder+"/model-{epoch:02d}-{val_accuracy:.2f}.hdf5"

checkpoint_tr = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_accuracy', verbose=1,
    save_best_only=False, save_weights_only=False,
    save_frequency=1)

CHECKPOINT_DIR_tr = "/media/SSD1TB/rezaei/Projects/GuidedCTCOCR/Data/" + 'Transformer' + "/"
CHECKPOINT_DIR_ctc = "/media/SSD1TB/rezaei/Projects/GuidedCTCOCR/Data/" + 'CTC' + "/"

os.makedirs(CHECKPOINT_DIR_tr, exist_ok=True)
os.makedirs(CHECKPOINT_DIR_ctc, exist_ok=True)

#os.makedirs(os.path.join(CHECKPOINT_DIR, "bestModel"), exist_ok=True)
#filepath1 = os.path.join(os.curdir, "saved_model")
# filepath='../saved_model/weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5'
filepath2 = 'weights-improvement-{epoch:02d}-{loss:.2f}.hdf5'
filepath_ctc  = os.path.join(CHECKPOINT_DIR_ctc , filepath2)
backup_ckpt_ctc = tf.keras.callbacks.BackupAndRestore(backup_dir=CHECKPOINT_DIR_ctc )
checkpoint_ctc =tf.keras.callbacks.ModelCheckpoint(filepath_ctc,
                                              verbose=1,
                                              save_best_only=True, monitor ="loss")
filepath_tr  = os.path.join(CHECKPOINT_DIR_tr , filepath2)
backup_ckpt_tr = tf.keras.callbacks.BackupAndRestore(backup_dir=filepath)#CHECKPOINT_DIR_tr)
# checkpoint_tr =tf.keras.callbacks.ModelCheckpoint(filepath_tr,
#                                               verbose=1,
#                                               save_best_only=True, monitor ="loss")
tbCallBack_ctc=tf.keras.callbacks.TensorBoard(log_dir='../my_logs_ctc', histogram_freq=1,  write_graph=True, write_images=True)
tbCallBack_tr=tf.keras.callbacks.TensorBoard(log_dir='../my_logs_tr', histogram_freq=1,  write_graph=True, write_images=True)



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=40000):#4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
