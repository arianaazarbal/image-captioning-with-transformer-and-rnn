import tensorflow as tf

try: from transformer import TransformerBlock, PositionalEncoding
except Exception as e: print(f"TransformerDecoder Might Not Work, as components failed to import:\n{e}")

########################################################################################

class RNNDecoder(tf.keras.layers.Layer):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # Now we will define image and word embedding, decoder, and classification layers
        # Define feed forward layer(s) to embed image features into a vector 
        self.image_embedding = tf.keras.layers.Dense(self.hidden_size)

        # Define english embedding layer:
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)

        # Define decoder layer that handles language and image context:     
        self.decoder = tf.keras.layers.GRU(self.hidden_size, return_sequences=True)

        # Define classification layer(s) (LOGIT OUTPUT)
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.vocab_size, activation='relu'),
            tf.keras.layers.Dense(self.vocab_size)])
        
    @tf.function
    def call(self, encoded_images, captions):
        # 1) Embed the encoded images into a vector of the correct dimension for initial state
        # 2) Pass your english sentance embeddings, and the image embeddings, to your decoder 
        # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
        encoded_images = self.image_embedding(encoded_images)
        text_embeddings = self.embedding(captions)
        logits = self.decoder(text_embeddings, initial_state=encoded_images)
        logits = self.classifier(logits)
        return logits


########################################################################################

class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.image_embedding = tf.keras.layers.Dense(self.hidden_size)

        # Define english embedding layer:
        self.encoding = PositionalEncoding(self.vocab_size, self.hidden_size, self.window_size)

        # Define decoder layer that handles language and image context:     
        self.decoder = TransformerBlock(self.hidden_size)
        # Define classification layer(s) (LOGIT OUTPUT)
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(self.vocab_size, activation='relu'),
            tf.keras.layers.Dense(self.vocab_size)])

    def call(self, encoded_images, captions):
        encoded_images = self.image_embedding(tf.expand_dims(encoded_images, 1))
        captions = self.encoding(captions)
        logits = self.decoder(captions, encoded_images)
        logits = self.classifier(logits)
        return logits
