import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
MAX_LENGTH = 40
EMBEDDING_DIM = 512
UNITS = 512

@st.cache_resource
def load_tokenizer_and_vocab():
    try:
        with open('vocab_coco.file', 'rb') as f:
            vocab = pickle.load(f)

        tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=None, 
            standardize=None,
            output_sequence_length=MAX_LENGTH,
            vocabulary=vocab
        )

        word2idx = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary()
        )

        idx2word = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary(),
            invert=True
        )

        return tokenizer, idx2word, len(vocab)
    except Exception as e:
        st.error(f"Error loading tokenizer/vocab: {e}")
        raise e


#model architecture
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")

    def call(self, x, training):
        x = self.layer_norm_1(x)
        x = self.dense(x)
        attn_output = self.attention(
            query=x, value=x, key=x, attention_mask=None, training=training)
        x = self.layer_norm_2(x + attn_output)
        return x


class Embeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(
            max_len, embed_dim, input_shape=(None, max_len))

    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]
        position_ids = tf.range(start=0, limit=length, delta=1)
        position_ids = tf.expand_dims(position_ids, axis=0)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        return token_embeddings + position_embeddings


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, units, num_heads, vocab_size):
        super().__init__()
        self.embedding = Embeddings(vocab_size, embed_dim, MAX_LENGTH)
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)
        self.out = tf.keras.layers.Dense(vocab_size, activation="softmax")
        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0)
        return tf.tile(mask, mult)

    def call(self, input_ids, encoder_output, training, mask=None):
        embeddings = self.embedding(input_ids)
        combined_mask = None
        padding_mask = None
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(embeddings)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        attn_output_1 = self.attention_1(
            query=embeddings, value=embeddings, key=embeddings,
            attention_mask=combined_mask, training=training)
        out_1 = self.layernorm_1(embeddings + attn_output_1)
        attn_output_2, attn_scores_2 = self.attention_2(
            query=out_1, value=encoder_output, key=encoder_output,
            attention_mask=padding_mask, training=training,return_attention_scores=True)
        out_2 = self.layernorm_2(out_1 + attn_output_2)
        self.last_attention_scores = attn_scores_2
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds


class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug

    def call(self, inputs, training=False):
        img, captions = inputs
        img_embed = self.cnn_model(img)
        img_encoded = self.encoder(img_embed, training=training)
        y_pred = self.decoder(captions, img_encoded, training=training, mask=tf.cast(captions != 0, tf.int32))
        return y_pred


#CNN model
def CNN_Encoder():
    inception_v3 = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    output = inception_v3.output
    output = tf.keras.layers.Reshape((-1, output.shape[-1]))(output)
    cnn_model = tf.keras.models.Model(inception_v3.input, output)
    return cnn_model


#Loading model
@st.cache_resource
def load_model(vocab_size):
    try:
        cnn_model = CNN_Encoder()
        encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
        decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8, vocab_size)
        caption_model = ImageCaptioningModel(cnn_model, encoder, decoder)

        # Build model with dummy input
        dummy_img = tf.zeros((1, 299, 299, 3))
        dummy_caption = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
        caption_model([dummy_img, dummy_caption], training=False)

        # Load weights
        caption_model.load_weights("model.h5")
        return caption_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e


#Preprocess image
def preprocess_image(image, target_size=(299, 299)):
    try:
        image = image.resize(target_size)
        image = np.array(image)
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = image.astype(np.float32)
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        raise e


#Generate caption
def generate_caption_from_embedding(model, img, tokenizer, idx2word):
    try:
        img_embed = model.cnn_model(img)
        img_encoded = model.encoder(img_embed, training=False)
 
        y_inp = '[start]'
        words = []
        attention_maps = []  # one entry per generated word
 
        for i in range(MAX_LENGTH - 1):
            tokenized = tokenizer([y_inp])[:, :-1]
            mask = tf.cast(tokenized != 0, tf.int32)
            pred = model.decoder(
                tokenized, img_encoded, training=False, mask=mask)
 
            # grab the attention weights the decoder just stashed
            # shape: (1, 8 heads, caption_len, 64 patches) -> average
            # over heads, take the row for the word we're generating now
            attn = model.decoder.last_attention_scores  # (1, 8, seq, 64)
            attn_this_word = tf.reduce_mean(attn[0, :, i, :], axis=0)  # (64,)
            attention_maps.append(attn_this_word.numpy())
 
            pred_idx = np.argmax(pred[0, i, :])
            pred_idx = tf.convert_to_tensor(pred_idx)
            pred_word = idx2word(pred_idx).numpy().decode('utf-8')
 
            if pred_word == '[end]':
                break
 
            words.append(pred_word)
            y_inp += ' ' + pred_word
 
        caption = ' '.join(words)
        return caption, words, attention_maps
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        raise e

import matplotlib.pyplot as plt
import matplotlib.cm as cm
 
def plot_attention_maps(image, words, attention_maps):
    """
    Renders one small subplot per generated word, showing the original
    image with a heatmap overlay of where the model 'looked' when it
    generated that word. attention_maps[i] is a flat 64-length array
    (8x8 grid of image patches) for word i.
    """
    image = image.resize((299, 299))
    image_np = np.array(image)
 
    num_words = len(words)
    cols = 4
    rows = int(np.ceil(num_words / cols))
 
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
 
    for i, (word, attn) in enumerate(zip(words, attention_maps)):
        # reshape the 64 flat scores back into the 8x8 grid of patches
        attn_grid = attn.reshape(8, 8)
        # upsample the 8x8 grid to 299x299 so it overlays the image cleanly
        attn_resized = tf.image.resize(
            attn_grid[..., np.newaxis], (299, 299), method='bilinear'
        ).numpy().squeeze()
        attn_resized = attn_resized / (attn_resized.max() + 1e-8)
 
        axes[i].imshow(image_np)
        axes[i].imshow(attn_resized, cmap=cm.jet, alpha=0.5)
        axes[i].set_title(word, fontsize=11)
        axes[i].axis('off')
 
    for j in range(num_words, len(axes)):
        axes[j].axis('off')
 
    plt.tight_layout()
    return fig


#Streamlit part
st.title("Image Captioning App")
st.write("Upload an image to generate a caption using the trained model!")

if st.button("Clear and Upload New Image"):
    st.session_state.image_uploaded = False
    st.session_state.caption = ""
    st.session_state.image = None
    st.rerun()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file is not None and not st.session_state.image_uploaded:
    try:
        image = Image.open(uploaded_file)
        st.session_state.image = image
        st.session_state.image_uploaded = True
        processed_image = preprocess_image(image)
        tokenizer, index_to_word, vocabulary_size = load_tokenizer_and_vocab()
        model = load_model(vocabulary_size)
        with st.spinner("Generating caption..."):
            caption, words, attention_maps = generate_caption_from_embedding(model, processed_image, tokenizer, index_to_word)
        st.session_state.caption = caption
        st.session_state.words = words
        st.session_state.attention_maps = attention_maps
    except Exception as e:
        st.error(f"Error processing image: {e}")

# Display image and caption
if st.session_state.image_uploaded and st.session_state.image:
    st.image(st.session_state.image, caption="Uploaded Image", use_column_width=True)
    if st.session_state.caption:
        st.success("**Generated Caption**:")
        st.markdown(f"**{st.session_state.caption}**")

        with st.expander("See what the model was looking at (attention maps)"):
            st.write(
                "Each tile shows the image with a heatmap of which region "
                "the model attended to most while generating that word."
            )
            fig = plot_attention_maps(
                st.session_state.image,
                st.session_state.words,
                st.session_state.attention_maps
            )
            st.pyplot(fig)
            plt.close(fig)
