"""NRC Emotion Lexicon mapping for Hamlet words.

Maps each word to emotion scores across 8 dimensions:
joy, sadness, fear, anger, trust, disgust, surprise, anticipation
"""

# Curated subset of NRC Emotion Lexicon for Hamlet Act 1 Scene 1 words
# Format: word -> {emotion: score (0-1)}
WORD_EMOTIONS = {
    # Common verbs & pronouns
    "there": {"anticipation": 0.5},
    "answer": {"anticipation": 0.6},
    "meet": {"trust": 0.6, "joy": 0.4},
    "hear": {"anticipation": 0.5, "surprise": 0.4},
    "see": {"anticipation": 0.4, "surprise": 0.3},

    # Positive/Trust/Joy
    "live": {"joy": 0.875, "trust": 0.875, "anticipation": 0.625},
    "welcome": {"joy": 0.875, "trust": 0.875},
    "relief": {"joy": 0.875, "trust": 0.75},
    "good": {"joy": 0.75, "trust": 0.75},
    "friend": {"trust": 0.875, "joy": 0.625},
    "haste": {"anticipation": 0.75},
    "approve": {"trust": 0.75, "joy": 0.6},

    # Negative/Fear/Sadness
    "fear": {"fear": 0.875, "sadness": 0.625},
    "dreaded": {"fear": 0.875, "sadness": 0.75},
    "apparition": {"fear": 0.875, "surprise": 0.75},
    "terrible": {"fear": 0.75, "anger": 0.625},
    "bitter": {"sadness": 0.875},
    "cold": {"sadness": 0.625, "fear": 0.5},
    "sick": {"sadness": 0.875, "fear": 0.625},

    # Surprise/Wonder
    "appear": {"surprise": 0.75, "anticipation": 0.625},
    "strange": {"surprise": 0.75},
    "sight": {"surprise": 0.625, "anticipation": 0.5},
    "look": {"surprise": 0.4, "anticipation": 0.3},

    # Trust/Safety
    "peace": {"trust": 0.875, "joy": 0.625},
    "honest": {"trust": 0.875},
    "carefully": {"trust": 0.6, "anticipation": 0.5},

    # Disgust/Anger
    "tush": {"disgust": 0.75, "anger": 0.625},
    "break": {"anger": 0.6, "disgust": 0.3},

    # Action/Anticipation
    "watch": {"anticipation": 0.75},
    "waiting": {"anticipation": 0.75},
    "speak": {"anticipation": 0.5},
    "come": {"anticipation": 0.75, "surprise": 0.4},
}


def get_emotion_vector(word: str) -> dict:
    """Get normalized emotion vector for a word.

    Returns dict with keys: joy, sadness, fear, anger, trust, disgust, surprise, anticipation
    All values in [0, 1], defaulting to 0.0 for unknown words.
    """
    emotions = {
        "joy": 0.0,
        "sadness": 0.0,
        "fear": 0.0,
        "anger": 0.0,
        "trust": 0.0,
        "disgust": 0.0,
        "surprise": 0.0,
        "anticipation": 0.0,
    }

    # Try exact match
    word_lower = word.lower().strip()
    if word_lower in WORD_EMOTIONS:
        emotions.update(WORD_EMOTIONS[word_lower])
    # Try without punctuation
    elif word_lower.rstrip(".,!?;:—-'") in WORD_EMOTIONS:
        emotions.update(WORD_EMOTIONS[word_lower.rstrip(".,!?;:—-'")])

    return emotions
