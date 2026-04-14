"""
Simulation configuration parameters.
"""

# ── World ─────────────────────────────────────────────────────────────────
WORLD_W   = 1.0
WORLD_H   = 1.0
WINDOW_PX = 1280
WINDOW_PY = 900

# ── Test-mode semantic strands ────────────────────────────────────────────
# Each list = one RNA molecule (all semantic words, no fillers).
# Words are from actual Hamlet text — their embeddings are meaningful.
TEST_MODE = True

TEST_STRANDS = [
    ["nobler", "mind", "suffer", "slings", "arrows", "outrageous", "fortune"],
    ["conscience", "cowards", "native", "hue", "resolution", "sicklied", "pale"],
    ["sleep", "perchance", "dream", "death", "rub", "dreams", "come"],
    ["piece", "work", "man", "noble", "reason", "infinite", "faculty"],
    ["night", "sweet", "prince", "angels", "flights", "sing", "rest"],
]

# Spawn row y-positions (one per strand), all centred at x=0.5
# Equal vertical spacing so all 5 chains are clearly visible and separated
SPAWN_Y_POSITIONS = [0.82, 0.64, 0.49, 0.33, 0.17]

# ── Smoker position (used only for glow init; spawns use SPAWN_Y_POSITIONS)
SMOKER_X = 0.5
SMOKER_Y = 0.04

# ── Capacity ──────────────────────────────────────────────────────────────
MAX_BEADS     = 200
MAX_STRANDS   = 10
MAX_STRAND_LEN = 60
MAX_EMIT_STRANDS = len(TEST_STRANDS)

# ── Bead ──────────────────────────────────────────────────────────────────
BEAD_RADIUS = 0.005
BEAD_MASS   = 1.0

# ── Bond (harmonic spring) ────────────────────────────────────────────────
BOND_K         = 900.0
# Rest length = BASE + CHAR_W * len(word_on_this_bond)
# CHAR_W ≈ avg character width in world units  (Georgia 16pt ≈ 8.5 px / 1280 px)
BOND_REST_BASE = 0.016   # gap padding in world units
BOND_REST_CHAR = 0.0072  # world units per character of the word on this bond
BOND_REST      = 0.050   # fallback scalar (not used in main path)
BOND_DAMP      = 0.5

# ── Bending stiffness ─────────────────────────────────────────────────────
# High → strands strongly prefer straight; semantic forces must fight this
BEND_K = 15.0

# ── Non-bonded LJ (intra-strand only) ────────────────────────────────────
LJ_EPSILON  = 0.0025
LJ_SIGMA    = 0.026
LJ_CUTOFF   = 0.25
COS_NEUTRAL = 0.10

ATTRACT_LINE_THRESHOLD = 0.12
ATTRACT_LINE_MAX_DIST  = 0.40

# ── Brownian / thermal ────────────────────────────────────────────────────
KT      = 0.00025
DT      = 0.0008
DAMPING = 0.968

# ── Rendering ─────────────────────────────────────────────────────────────
FONT_PREFERENCE    = ["Georgia", "Palatino", "Garamond", "Times New Roman",
                      "Baskerville", "Book Antiqua", "serif"]
FONT_SIZE_SEMANTIC = 16
FONT_SIZE_FILLER   = 11   # not used (fillers fully invisible)

BG_COLOR           = (10, 5, 20)
HALO_RINGS         = 4
HALO_ALPHA_BASE    = 52
HALO_SCALE         = 1.85

# Tiny pivot dot at each joint (set to 0 to hide)
PIVOT_DOT_RADIUS = 2
PIVOT_DOT_COLOR  = (55, 55, 80)

# ── Embedding ─────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_CACHE = "cache/embeddings.pkl"

# ── Filler words (stripped from test strands; kept for full-corpus mode) ──
FILLER_WORDS = {
    "a", "an", "the",
    "and", "but", "or", "nor", "for", "yet", "so",
    "to", "of", "in", "on", "at", "by", "with", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "among", "upon", "about", "against", "along", "over", "under",
    "i", "me", "my", "mine", "we", "our", "ours", "us",
    "you", "your", "yours",
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "they", "them", "their", "theirs",
    "thou", "thee", "thy", "thine", "ye",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should",
    "may", "might", "must", "can", "could",
    "that", "this", "these", "those", "which", "who", "whom", "whose",
    "what", "when", "where", "how", "why", "if", "then", "than",
    "not", "no", "more", "so", "such", "all", "there",
}
