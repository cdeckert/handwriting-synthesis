import logging
import os
import pprint as pp
from collections import deque
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


class TFBaseModel:
    """Base class providing a high level training loop for TensorFlow 2 models.

    Subclasses are responsible for implementing ``build_model`` and
    ``calculate_loss``. The base class handles logging, checkpointing, early
    stopping and learning rate restarts.
    """

    def __init__(
        self,
        reader=None,
        batch_sizes=(128,),
        num_training_steps=20000,
        learning_rates=(0.01,),
        beta1_decays=(0.99,),
        optimizer="adam",
        grad_clip=5.0,
        regularization_constant=0.0,
        keep_prob=1.0,
        patiences=(3000,),
        warm_start_init_step=0,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=100,
        log_interval=20,
        logging_level=logging.INFO,
        loss_averaging_window=100,
        validation_batch_size=64,
        log_dir="logs",
        checkpoint_dir="checkpoints",
        prediction_dir="predictions",
    ):
        if enable_parameter_averaging:
            raise ValueError("Parameter averaging is not supported in the TF2 implementation.")

        assert len(batch_sizes) == len(learning_rates) == len(patiences)
        self.batch_sizes = list(batch_sizes)
        self.learning_rates = list(learning_rates)
        self.beta1_decays = list(beta1_decays)
        self.patiences = list(patiences)
        self.num_restarts = len(self.batch_sizes) - 1
        self.restart_idx = 0
        self.update_train_params()

        self.reader = reader
        self.num_training_steps = num_training_steps
        self.optimizer_name = optimizer
        self.grad_clip = grad_clip
        self.regularization_constant = regularization_constant
        self.warm_start_init_step = warm_start_init_step
        self.keep_prob_scalar = keep_prob
        self.min_steps_to_checkpoint = min_steps_to_checkpoint
        self.log_interval = log_interval
        self.loss_averaging_window = loss_averaging_window
        self.validation_batch_size = validation_batch_size

        self.log_dir = log_dir
        self.logging_level = logging_level
        self.prediction_dir = prediction_dir
        self.checkpoint_dir = checkpoint_dir

        self.metrics: Dict[str, float] = {}
        self.early_stopping_metric = "val_loss"

        self.init_logging(self.log_dir)
        logging.info("\nnew run with parameters:\n{}".format(pp.pformat(self.__dict__)))

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model = self.build_model()
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.optimizer = self.get_optimizer(self.learning_rate, self.beta1_decay)

        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            global_step=self.global_step,
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_dir, max_to_keep=1
        )
        if self.warm_start_init_step:
            self.restore()

    def update_train_params(self):
        self.batch_size = self.batch_sizes[self.restart_idx]
        self.learning_rate = self.learning_rates[self.restart_idx]
        self.beta1_decay = self.beta1_decays[self.restart_idx]
        self.early_stopping_steps = self.patiences[self.restart_idx]

    def build_model(self):  # pragma: no cover - to be implemented by subclasses
        raise NotImplementedError

    def calculate_loss(self, batch: Dict[str, np.ndarray], training: bool = True) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        raise NotImplementedError

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    def fit(self):
        best_validation_metric = float("inf")
        best_validation_step = 0

        train_generator = self.reader.train_batch_generator(self.batch_size)
        val_generator = self.reader.val_batch_generator(self.validation_batch_size)

        train_loss_history = deque(maxlen=self.loss_averaging_window)
        val_loss_history = deque(maxlen=self.loss_averaging_window)
        train_time_history = deque(maxlen=self.loss_averaging_window)
        val_time_history = deque(maxlen=self.loss_averaging_window)
        metric_histories = {metric_name: deque(maxlen=self.loss_averaging_window) for metric_name in self.metrics}

        step = int(self.global_step.numpy())

        while step < self.num_training_steps:
            import time

            # Validation
            val_start = time.time()
            val_batch = next(val_generator)
            val_loss, val_metrics = self.calculate_loss(val_batch, training=False)
            val_loss = float(val_loss.numpy())
            val_loss_history.append(val_loss)
            val_time_history.append(time.time() - val_start)

            for metric_name in self.metrics:
                metric_value = val_metrics.get(metric_name)
                if metric_value is not None:
                    metric_histories[metric_name].append(float(metric_value.numpy()))

            # Training
            train_start = time.time()
            train_batch = next(train_generator)
            train_loss, train_metrics = self.train_step(train_batch)
            train_loss_history.append(train_loss)
            train_time_history.append(time.time() - train_start)

            step = int(self.global_step.numpy())

            if step % self.log_interval == 0:
                avg_train_loss = sum(train_loss_history) / max(len(train_loss_history), 1)
                avg_val_loss = sum(val_loss_history) / max(len(val_loss_history), 1)
                avg_train_time = sum(train_time_history) / max(len(train_time_history), 1)
                avg_val_time = sum(val_time_history) / max(len(val_time_history), 1)

                metric_log = (
                    "[[step {:>8}]]     "
                    "[[train {:>4}s]]     loss: {:<12}     "
                    "[[val {:>4}s]]     loss: {:<12}     "
                ).format(
                    step,
                    round(avg_train_time, 4),
                    round(avg_train_loss, 8),
                    round(avg_val_time, 4),
                    round(avg_val_loss, 8),
                )

                early_stopping_metric = avg_val_loss
                for metric_name in self.metrics:
                    if metric_histories[metric_name]:
                        metric_val = sum(metric_histories[metric_name]) / len(metric_histories[metric_name])
                        metric_log += f"{metric_name}: {round(metric_val, 4)}     "
                        if metric_name == self.early_stopping_metric:
                            early_stopping_metric = metric_val

                logging.info(metric_log)

                if early_stopping_metric < best_validation_metric:
                    best_validation_metric = early_stopping_metric
                    best_validation_step = step
                    if step > self.min_steps_to_checkpoint:
                        self.save(step)

                if step - best_validation_step > self.early_stopping_steps:
                    if self.num_restarts is None or self.restart_idx >= self.num_restarts:
                        logging.info(
                            "best validation {} of {} at training step {}".format(
                                self.early_stopping_metric,
                                best_validation_metric,
                                best_validation_step,
                            )
                        )
                        logging.info("early stopping - ending training.")
                        return

                    if self.restart_idx < self.num_restarts:
                        logging.info("validation plateau reached - restarting with new hyperparameters")
                        self.restore(best_validation_step)
                        self.restart_idx += 1
                        self.update_train_params()
                        self.optimizer = self.get_optimizer(self.learning_rate, self.beta1_decay)
                        self.checkpoint = tf.train.Checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            global_step=self.global_step,
                        )
                        self.manager = tf.train.CheckpointManager(
                            self.checkpoint, directory=self.checkpoint_dir, max_to_keep=1
                        )
                        train_generator = self.reader.train_batch_generator(self.batch_size)

        if step <= self.min_steps_to_checkpoint:
            self.save(step)
        logging.info("num_training_steps reached - ending training")

    def train_step(self, batch: Dict[str, np.ndarray]):
        with tf.GradientTape() as tape:
            loss_tensor, metrics = self.calculate_loss(batch, training=True)
            loss_value = loss_tensor
            if self.regularization_constant:
                l2_norm = tf.add_n(
                    [tf.sqrt(tf.reduce_sum(tf.square(param))) for param in self.trainable_variables]
                )
                loss_value = loss_value + self.regularization_constant * l2_norm

        gradients = tape.gradient(loss_value, self.trainable_variables)
        clipped_gradients = []
        for grad in gradients:
            if grad is None:
                clipped_gradients.append(None)
            else:
                clipped_gradients.append(tf.clip_by_value(grad, -self.grad_clip, self.grad_clip))

        self.optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables))
        self.global_step.assign_add(1)

        return float(loss_tensor.numpy()), metrics

    def save(self, step=None):
        if not os.path.isdir(self.checkpoint_dir):
            logging.info("creating checkpoint directory %s", self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = self.manager.save(checkpoint_number=step)
        logging.info("saved checkpoint to %s", path)

    def restore(self, step=None):
        if not os.path.isdir(self.checkpoint_dir):
            raise FileNotFoundError("checkpoint directory does not exist")

        if step is None:
            ckpt_path = self.manager.latest_checkpoint
        else:
            ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt-{step}")
        if ckpt_path is None:
            raise FileNotFoundError("no checkpoint found to restore")
        logging.info("restoring model parameters from %s", ckpt_path)
        self.checkpoint.restore(ckpt_path).expect_partial()

    def init_logging(self, log_dir):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_file = f"log_{date_str}.txt"

        logging.basicConfig(
            filename=os.path.join(log_dir, log_file),
            level=self.logging_level,
            format="[[%(asctime)s]] %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )
        logging.getLogger().addHandler(logging.StreamHandler())

    def get_optimizer(self, learning_rate, beta1_decay):
        if self.optimizer_name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1_decay)
        if self.optimizer_name == "gd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        if self.optimizer_name == "rms":
            return tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate, rho=beta1_decay, momentum=0.9
            )
        raise ValueError("optimizer must be adam, gd, or rms")
