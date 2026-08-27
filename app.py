import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import io
import base64
import matplotlib.cm as cm
import streamlit.components.v1 as components
from huggingface_hub import hf_hub_download

# Model weights (~465MB) and vocab live in an HF Hub MODEL repo, not a Space -
# GitHub rejects files over 100MB, and model-repo storage is unaffected by
# the Spaces compute paywall change. Update this after you create the repo.
HF_REPO_ID = "Akku0902/image-captioning-weights"

MAX_LENGTH = 40
EMBEDDING_DIM = 512
UNITS = 512
BEAM_WIDTH = 3
LENGTH_PENALTY = 0.7
MAX_PINNED_WORDS = 8  # keep the annotated image readable; full list still shown below

WEIGHTS_FILENAME = "model.weights.h5"  # Keras 3 requires this exact suffix
BLEU_VAL_SCORE = "0.0796"

# ---------------------------------------------------------------------------
# Design tokens - light, pale-sage direction (no dark backgrounds anywhere)
# ---------------------------------------------------------------------------
INK = "#F1F3EE"       # page background - pale sage-white
PANEL = "#FFFFFF"      # card background - pure white, floats on the page
BORDER = "#DEE3D9"     # light sage-grey border
ACCENT = "#3F6B4A"     # deep leaf green
BONE = "#1E211C"       # primary text - near-black
MUTED = "#68705F"      # secondary text - muted sage-grey

st.set_page_config(page_title="Image Captioning - Transformer", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background-color: {INK}; color: {BONE}; }}

h1, h2, h3 {{ font-family: 'Newsreader', serif !important; color: {BONE} !important; }}

.eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.4rem;
}}

.hero-title {{
    font-family: 'Newsreader', serif;
    font-weight: 600;
    font-size: 2.3rem;
    line-height: 1.2;
    color: {BONE};
    margin-bottom: 0.5rem;
}}

.hero-sub {{
    color: {MUTED};
    font-size: 1rem;
    max-width: 640px;
    margin-bottom: 1.6rem;
}}

.meta-pill {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: {MUTED};
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 3px 11px;
    margin: 0 6px 6px 0;
}}

.word-chip {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: {ACCENT};
    background-color: rgba(63, 107, 74, 0.08);
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 2px 8px;
    margin-top: 6px;
}}

[data-testid="stFileUploaderDropzone"] {{
    background-color: {PANEL} !important;
    border: 1.5px dashed {BORDER} !important;
    border-radius: 12px !important;
}}

.stButton > button {{
    background-color: transparent;
    color: {ACCENT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
}}
.stButton > button:hover {{
    border-color: {ACCENT};
    color: {BONE};
    background-color: rgba(63, 107, 74, 0.06);
}}

[data-testid="stImage"] img {{
    border-radius: 8px;
    border: 1px solid {BORDER};
}}

/* nav bar */
.nav-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 14px;
    margin-bottom: 22px;
    border-bottom: 1px solid {BORDER};
}}
.nav-logo {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: {BONE};
    letter-spacing: 0.04em;
}}
.nav-logo span {{ color: {ACCENT}; }}
.nav-link {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: {MUTED};
    text-decoration: none;
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 5px 13px;
}}
.nav-link:hover {{ color: {BONE}; border-color: {ACCENT}; }}

/* two-tone hero headline */
.hero-title-2 {{
    font-family: 'Newsreader', serif;
    font-weight: 600;
    font-size: 2.6rem;
    line-height: 1.18;
    color: {BONE};
    margin-bottom: 0.6rem;
}}
.hero-title-2 .accent {{ color: {ACCENT}; }}

/* CTA pill links (real destinations, styled like buttons) */
.cta-row {{ margin: 1.1rem 0 1.8rem 0; }}
.cta-pill {{
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    text-decoration: none;
    border-radius: 999px;
    padding: 9px 18px;
    margin-right: 10px;
}}
.cta-pill.solid {{ background-color: {ACCENT}; color: {INK}; }}
.cta-pill.outline {{ color: {BONE}; border: 1px solid {BORDER}; }}
.cta-pill.outline:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

/* browser-style dot strip atop the framed mockup */
.frame-strip {{
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 10px;
}}
.frame-dot {{ width: 7px; height: 7px; border-radius: 50%; background: {MUTED}; opacity: 0.4; }}
.frame-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: {MUTED};
    margin-left: 6px;
}}

/* framed mockup container (native Streamlit bordered container) */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {PANEL} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 16px !important;
    padding: 6px;
}}

/* bottom footer pill row */
.footer-row {{
    display: flex;
    gap: 10px;
    margin-top: 1.8rem;
    padding-top: 1.2rem;
    border-top: 1px solid {BORDER};
}}

hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("image_uploaded", False),
    ("caption", ""),
    ("image", None),
    ("words", []),
    ("attention_maps", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Tokenizer / vocab
# ---------------------------------------------------------------------------
@st.cache_resource
def load_tokenizer_and_vocab():
    vocab_path = hf_hub_download(repo_id=HF_REPO_ID, filename="vocab_coco.file")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=None,
        standardize=None,
        output_sequence_length=MAX_LENGTH,
        vocabulary=vocab
    )
    idx2word = tf.keras.layers.StringLookup(
        mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True
    )
    return tokenizer, idx2word, len(vocab)


# ---------------------------------------------------------------------------
# Model architecture (unchanged from training)
# ---------------------------------------------------------------------------
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
            attention_mask=padding_mask, training=training, return_attention_scores=True)
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


def CNN_Encoder():
    inception_v3 = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    output = inception_v3.output
    output = tf.keras.layers.Reshape((-1, output.shape[-1]))(output)
    return tf.keras.models.Model(inception_v3.input, output)


@st.cache_resource
def load_model(vocab_size):
    cnn_model = CNN_Encoder()
    encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
    decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8, vocab_size)
    caption_model = ImageCaptioningModel(cnn_model, encoder, decoder)

    dummy_img = tf.zeros((1, 299, 299, 3))
    dummy_caption = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
    caption_model([dummy_img, dummy_caption], training=False)

    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=WEIGHTS_FILENAME)
    caption_model.load_weights(weights_path)
    return caption_model


def preprocess_image(image, target_size=(299, 299)):
    image = image.resize(target_size)
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = image.astype(np.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return np.expand_dims(image, axis=0)


# ---------------------------------------------------------------------------
# Beam search generation, with attention captured per winning-beam step
# ---------------------------------------------------------------------------
def generate_caption_beam_search(model, img, tokenizer, idx2word,
                                  beam_width=BEAM_WIDTH, max_len=MAX_LENGTH,
                                  length_penalty=LENGTH_PENALTY):
    img_embed = model.cnn_model(img)
    img_encoded = model.encoder(img_embed, training=False)

    beams = [('[start]', 0.0, False, [])]

    for step in range(max_len - 1):
        all_candidates = []
        for seq, score, finished, attn_list in beams:
            if finished:
                all_candidates.append((seq, score, finished, attn_list))
                continue

            tokenized = tokenizer([seq])[:, :-1]
            mask = tf.cast(tokenized != 0, tf.int32)
            pred = model.decoder(tokenized, img_encoded, training=False, mask=mask)

            attn = model.decoder.last_attention_scores
            attn_this_step = tf.reduce_mean(attn[0, :, step, :], axis=0).numpy()

            probs = pred[0, step, :].numpy()
            log_probs = np.log(probs + 1e-10)
            top_k_idx = np.argsort(log_probs)[-beam_width:]

            for idx in top_k_idx:
                word = idx2word(tf.convert_to_tensor(idx)).numpy().decode('utf-8')
                new_score = score + log_probs[idx]
                if word == '[end]':
                    all_candidates.append((seq, new_score, True, attn_list))
                else:
                    all_candidates.append(
                        (seq + ' ' + word, new_score, False, attn_list + [attn_this_step])
                    )

        def normalized_score(cand):
            seq, score, finished, _ = cand
            length = max(len(seq.split()), 1)
            return score / (length ** length_penalty)

        beams = sorted(all_candidates, key=normalized_score, reverse=True)[:beam_width]
        if all(f for _, _, f, _ in beams):
            break

    best_seq, _, _, best_attn = beams[0]
    words = best_seq.replace('[start] ', '').split()
    return ' '.join(words), words, best_attn


# ---------------------------------------------------------------------------
# Attention -> pinned label positions on the image
# ---------------------------------------------------------------------------
def attention_peak_points(words, attn_maps):
    """For each word, find the (x%, y%) of peak attention in the 8x8 patch grid."""
    points = []
    for word, attn in zip(words, attn_maps):
        idx = int(np.argmax(attn))
        row, col = divmod(idx, 8)
        x_pct = (col + 0.5) / 8 * 100
        y_pct = (row + 0.5) / 8 * 100
        points.append((word, x_pct, y_pct))
    return points


def layout_pinned_labels(points, min_dy=9, step=7, max_attempts=20):
    """Nudge label y-position down when two labels would collide, so a dashed
    leader line connects the true attention point to a readable label.
    Bounded attempts: if many words cluster on the same patch, we accept a
    little overlap rather than looping forever trying to fully separate them."""
    ordered = sorted(points, key=lambda p: p[2])  # by y ascending
    placed = []
    results = []
    for word, x, y in ordered:
        label_y = y
        for _ in range(max_attempts):
            collision = any(abs(label_y - py) < min_dy for _, py in placed)
            if not collision or label_y >= 94:
                break
            label_y = min(label_y + step, 94)
        placed.append((x, label_y))
        results.append((word, x, y, label_y))
    return results


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_pinned_image_html(image, words, attn_maps, bleu_val=BLEU_VAL_SCORE):
    pinned_words = words[:MAX_PINNED_WORDS]
    pinned_attn = attn_maps[:MAX_PINNED_WORDS]
    points = attention_peak_points(pinned_words, pinned_attn)
    labeled = layout_pinned_labels(points)

    b64 = pil_to_base64(image.convert("RGB"))

    markers_html = ""
    for word, x, y, label_y in labeled:
        markers_html += f'<div style="position:absolute; left:{x:.1f}%; top:{y:.1f}%; width:6px; height:6px; margin:-3px 0 0 -3px; border-radius:50%; background:{ACCENT}; z-index:3;"></div>'
        if abs(label_y - y) > 2:
            markers_html += f'<div style="position:absolute; left:{x:.1f}%; top:{y:.1f}%; width:1px; height:{abs(label_y - y):.1f}%; border-left:1px dashed {MUTED}; z-index:2;"></div>'
        label_top = max(min(label_y, 94), 2)
        markers_html += (
            f'<div style="position:absolute; left:{x:.1f}%; top:{label_top:.1f}%; '
            f'transform:translate(-50%, -50%); font-family:\'IBM Plex Mono\',monospace; '
            f'font-size:11px; color:{BONE}; background:rgba(18,25,43,0.85); '
            f'border:1px solid {BORDER}; border-radius:5px; padding:1px 6px; z-index:4; '
            f'white-space:nowrap;">{word}</div>'
        )

    extra_note = ""
    if len(words) > MAX_PINNED_WORDS:
        extra_note = f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:11px; color:{MUTED}; margin-top:6px;">+{len(words) - MAX_PINNED_WORDS} more words - full breakdown below</div>'

    html = f"""
    <div style="background:{INK}; border-radius:14px; padding:16px;">
      <div class="eyebrow">encoder&ndash;decoder &middot; trained from scratch</div>
      <div style="position:relative; display:inline-block; width:100%;">
        <div style="position:absolute; top:6px; left:6px; width:14px; height:14px; border-top:1px solid {MUTED}; border-left:1px solid {MUTED}; z-index:5;"></div>
        <div style="position:absolute; bottom:6px; right:6px; width:14px; height:14px; border-bottom:1px solid {MUTED}; border-right:1px solid {MUTED}; z-index:5;"></div>
        <div style="position:absolute; top:-10px; right:6px; background:{ACCENT}; color:{INK}; font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:500; border-radius:6px; padding:4px 9px; z-index:6;">BLEU-4 (val) &middot; {bleu_val}</div>
        <div style="position:absolute; top:-10px; left:6px; background:{PANEL}; border:1px solid {BORDER}; color:{BONE}; font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:500; border-radius:6px; padding:4px 9px; z-index:6;">beam width 3</div>
        <div style="position:relative; border:1px solid {BORDER}; border-radius:8px; overflow:hidden;">
          <img src="data:image/png;base64,{b64}" style="width:100%; display:block;" />
          {markers_html}
        </div>
      </div>
      {extra_note}
    </div>
    """
    return html


def render_attention_tile(image_299, attn_flat_64):
    """Full per-word heatmap, for the detailed expander (more complete than
    the single peak-pin shown on the main image)."""
    attn_grid = attn_flat_64.reshape(8, 8)
    attn_resized = tf.image.resize(
        attn_grid[..., np.newaxis], (299, 299), method='bilinear'
    ).numpy().squeeze()
    attn_resized = attn_resized / (attn_resized.max() + 1e-8)

    heat = cm.jet(attn_resized)[..., :3]
    base = np.array(image_299).astype(np.float32) / 255.0
    if base.shape[-1] == 4:
        base = base[..., :3]

    blended = np.clip(0.55 * base + 0.45 * heat, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))


def typewriter_reveal(text, height=70):
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("</", "<\\/")
    components.html(f"""
    <div style="font-family:'Newsreader',serif; font-weight:500; font-size:1.9rem; color:{BONE}; line-height:1.3;">
      <span id="tw"></span><span style="color:{ACCENT}">&#9612;</span>
    </div>
    <script>
      const text = `{safe_text}`;
      const el = document.getElementById("tw");
      let i = 0;
      function tick() {{
        if (i <= text.length) {{
          el.textContent = text.slice(0, i);
          i++;
          setTimeout(tick, 28);
        }}
      }}
      tick();
    </script>
    """, height=height)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
GITHUB_URL = "https://github.com/Akankshaaa09/Image-Captioning-using-Transformers"
HF_MODEL_URL = f"https://huggingface.co/{HF_REPO_ID}"

st.markdown(
    f'<div class="nav-row">'
    f'<div class="nav-logo">IC<span>.</span>TRANSFORMER</div>'
    f'<a class="nav-link" href="{GITHUB_URL}" target="_blank">GitHub &#8599;</a>'
    f'</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="eyebrow">ENCODER&ndash;DECODER &middot; TRAINED FROM SCRATCH</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title-2">See what the <span class="accent">model sees.</span></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-sub">A transformer trained from scratch on COCO - InceptionV3 encoder, '
    '8-head cross-attention decoder - describes your image, then pins each word to exactly '
    'where it looked.</div>',
    unsafe_allow_html=True
)
st.markdown(
    f'<div class="cta-row">'
    f'<a class="cta-pill solid" href="#try-it">Try it below &darr;</a>'
    f'<a class="cta-pill outline" href="{HF_MODEL_URL}" target="_blank">Model card &#8599;</a>'
    f'</div>',
    unsafe_allow_html=True
)

EXAMPLE_IMAGES = [
    ("coffee", "coffee.jpg"),
    ("dog", "dog.jpg"),
    ("tennis", "tennis.jpg"),
]


def run_captioning(image):
    st.session_state.image = image
    st.session_state.image_uploaded = True

    processed_image = preprocess_image(image)
    tokenizer, idx2word, vocab_size = load_tokenizer_and_vocab()
    model = load_model(vocab_size)

    with st.spinner("Running beam search (width 3)..."):
        caption, words, attention_maps = generate_caption_beam_search(
            model, processed_image, tokenizer, idx2word
        )
    st.session_state.caption = caption
    st.session_state.words = words
    st.session_state.attention_maps = attention_maps


st.markdown('<div id="try-it"></div>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(
        '<div class="frame-strip">'
        '<div class="frame-dot"></div><div class="frame-dot"></div><div class="frame-dot"></div>'
        '<div class="frame-label">live demo</div>'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Drop an image, or click to browse",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )

    st.markdown(
        f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:11px; color:{MUTED}; margin:10px 0 4px 0;">OR TRY AN EXAMPLE</div>',
        unsafe_allow_html=True
    )
    example_cols = st.columns(len(EXAMPLE_IMAGES))
    for col, (label, path) in zip(example_cols, EXAMPLE_IMAGES):
        with col:
            try:
                st.image(path, use_container_width=True)
            except Exception:
                st.markdown(f'<div style="color:{MUTED}; font-size:11px;">missing: {path}</div>', unsafe_allow_html=True)
            if st.button(label, key=f"example_{label}"):
                try:
                    example_image = Image.open(path)
                except FileNotFoundError:
                    st.warning(f"'{path}' isn't in the repo yet — upload it to your examples/ folder on GitHub, then try again.")
                    example_image = None
                if example_image is not None:
                    run_captioning(example_image)
                    st.rerun()

    if st.button("Clear and upload new image"):
        st.session_state.image_uploaded = False
        st.session_state.caption = ""
        st.session_state.image = None
        st.session_state.words = []
        st.session_state.attention_maps = []
        st.rerun()

    if uploaded_file is not None and not st.session_state.image_uploaded:
        image = Image.open(uploaded_file)
        run_captioning(image)

    if st.session_state.image_uploaded and st.session_state.image and st.session_state.caption:
        pinned_html = render_pinned_image_html(
            st.session_state.image, st.session_state.words, st.session_state.attention_maps
        )
        st.markdown(pinned_html, unsafe_allow_html=True)

        typewriter_reveal(st.session_state.caption)

        st.markdown(
            f'<span class="meta-pill">beam search &middot; w3</span>'
            f'<span class="meta-pill">InceptionV3 encoder</span>'
            f'<span class="meta-pill">vocab 11,691</span>',
            unsafe_allow_html=True
        )

        with st.expander("Full attention detail (per-word heatmaps)"):
            st.caption(
                "The pins above show each word's single peak attention point. "
                "These tiles show the full heatmap each pin was simplified from."
            )
            image_299 = st.session_state.image.resize((299, 299))
            words = st.session_state.words
            attn_maps = st.session_state.attention_maps

            tiles_per_row = 4
            for row_start in range(0, len(words), tiles_per_row):
                row_words = words[row_start:row_start + tiles_per_row]
                row_attn = attn_maps[row_start:row_start + tiles_per_row]
                cols = st.columns(tiles_per_row)
                for col, word, attn in zip(cols, row_words, row_attn):
                    with col:
                        tile = render_attention_tile(image_299, attn)
                        st.image(tile, use_container_width=True)
                        st.markdown(f'<span class="word-chip">{word}</span>', unsafe_allow_html=True)

st.markdown(
    f'<div class="footer-row">'
    f'<a class="cta-pill outline" href="{GITHUB_URL}" target="_blank">View source &#8599;</a>'
    f'<a class="cta-pill outline" href="{HF_MODEL_URL}" target="_blank">Model weights &#8599;</a>'
    f'</div>',
    unsafe_allow_html=True
)
