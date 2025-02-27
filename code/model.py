import numpy as np
import tensorflow as tf

class ImageCaptionModel(tf.keras.Model):

    def __init__(self, decoder, **kwargs):
        super().__init__(**kwargs)
        self.decoder = decoder

    def get_config(self):
        return {"decoder": self.decoder} ## specific to ImageCaptionModel

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def call(self, encoded_images, captions):
        return self.decoder(encoded_images, captions)  

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]


    def train(self, train_captions, train_image_features, padding_index, batch_size=30):
        """
        Runs through one epoch - all training examples.

        :param model: the initialized model to use for forward and backward pass
        :param train_captions: train data captions (all data for training) 
        :param train_images: train image features (all data for training) 
        :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
        :return: None
        """
        num_batches = int(len(train_captions) / batch_size)

        total_loss = total_seen = total_correct = 0

        indices = np.arange(train_captions.shape[0])
        np.random.shuffle(indices)

        train_image_features = tf.gather(train_image_features, indices)
        train_captions = tf.gather(train_captions, indices)

        for index, end in enumerate(range(batch_size, len(train_captions)+1, batch_size)):

            # NOTE: 
            # - The captions passed to the decoder should have the last token in the window removed:
            #	 [<START> student working on homework <STOP>] --> [<START> student working on homework]
            #
            # - When computing loss, the decoder labels should have the first word removed:
            #	 [<START> student working on homework <STOP>] --> [student working on homework <STOP>]

            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            batch_image_features = train_image_features[start:end, :]
            decoder_input = train_captions[start:end, :-1]
            decoder_labels = train_captions[start:end, 1:]

            ## Perform a training forward pass. Make sure to factor out irrelevant labels.
            with tf.GradientTape() as tape:
                probs = self.call(batch_image_features, decoder_input)
                mask = decoder_labels != padding_index
                num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                loss = self.loss_function(probs, decoder_labels, mask)
                accuracy = self.accuracy_function(probs, decoder_labels, mask)
            grads = tape.gradient(loss, self.trainable_variables)
            grads_and_vars = zip(grads, self.trainable_variables)
            self.optimizer.apply_gradients(grads_and_vars)

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Training epoch {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')
        
        print()
        
        return avg_loss, avg_acc, avg_prp

    def test(self, test_captions, test_image_features, padding_index, batch_size=30):
        """
        DO NOT CHANGE; Use as inspiration

        Runs through one epoch - all testing examples.

        :param model: the initilized model to use for forward and backward pass
        :param test_captions: test caption data (all data for testing) of shape (num captions,20)
        :param test_image_features: test image feature data (all data for testing) of shape (num captions,1000)
        :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
        :returns: perplexity of the test set, per symbol accuracy on test set
        """
        num_batches = int(len(test_captions) / batch_size)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

            # NOTE: 
            # - The captions passed to the decoder should have the last token in the window removed:
            #	 [<START> student working on homework <STOP>] --> [<START> student working on homework]
            #
            # - When computing loss, the decoder labels should have the first word removed:
            #	 [<START> student working on homework <STOP>] --> [student working on homework <STOP>]

            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            batch_image_features = test_image_features[start:end, :]
            decoder_input = test_captions[start:end, :-1]
            decoder_labels = test_captions[start:end, 1:]

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            probs = self(batch_image_features, decoder_input)
            mask = decoder_labels != padding_index
            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            loss = self.loss_function(probs, decoder_labels, mask)
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()
        
        return avg_prp, avg_acc
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        decoder_config = config.pop("decoder")
        decoder = tf.keras.utils.deserialize_keras_object(decoder_config)
        return cls(decoder, **config)


def accuracy_function(prbs, labels, mask):
    """
    DO NOT CHANGE

    Computes the batch accuracy

    :param prbs:  float tensor, word prediction probabilities [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]
    :param labels:  integer tensor, word prediction labels [BATCH_SIZE x WINDOW_SIZE]
    :param mask:  tensor that acts as a padding mask [BATCH_SIZE x WINDOW_SIZE]
    :return: scalar tensor of accuracy of the batch between 0 and 1
    """
    correct_classes = tf.argmax(prbs, axis=-1) == labels
    accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
    return accuracy


def loss_function(prbs, labels, mask):
    """
    DO NOT CHANGE

    Calculates the model cross-entropy loss after one forward pass
    Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

    :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
    :param labels:  integer tensor, word prediction labels [batch_size x window_size]
    :param mask:  tensor that acts as a padding mask [batch_size x window_size]
    :return: the loss of the model as a tensor
    """
    masked_labs = tf.boolean_mask(labels, mask)
    masked_prbs = tf.boolean_mask(prbs, mask)
    scce = tf.keras.losses.sparse_categorical_crossentropy(masked_labs, masked_prbs, from_logits=True)
    loss = tf.reduce_sum(scce)
    return loss