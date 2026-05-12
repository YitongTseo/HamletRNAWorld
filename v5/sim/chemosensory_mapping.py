"""Map word emotions to chemosensory neuron activations.

In C. elegans, chemosensory neurons come in bilateral pairs and functional groups:
- ASEL/ASER: Bilateral pair responding to opposite valences (attractive vs. repulsive)
- AWAL/AWAR: Bilateral pair for appetitive responses
- AWBL/AWBR: Bilateral pair for additional chemical sensing
- ASI/ASJ: Feeding-related sensing (asymmetric)
- AWC: CO2 detection
- ASG/ASH: Touch/chemical integration

We map emotions to these neurons based on valence and arousal:
- Left-side neurons (L): Positive/approach emotions (joy, trust, anticipation)
- Right-side neurons (R): Negative/avoid emotions (sadness, disgust, fear)
- Central neurons: Intensity/novelty (anger, surprise)
"""

from __future__ import annotations

# Mapping emotions to chemosensory neuron activation patterns
EMOTION_TO_NEURONS = {
    # Positive/Approach emotions → Left-side, attractive cues
    "joy": {
        "ASEL": 0.9,      # Strong attractive signal (left)
        "AWAL": 0.85,     # Appetitive left
        "AWBL": 0.7,
    },
    "trust": {
        "ASEL": 0.75,     # Moderate attractive
        "AWAL": 0.8,
        "AWC": 0.5,       # Safe environment
    },
    "anticipation": {
        "AWAR": 0.6,      # Right-side expectation
        "AWAL": 0.7,      # Left-side seeking
        "AWCL": 0.5,
    },

    # Negative/Avoid emotions → Right-side, repulsive cues
    "sadness": {
        "ASER": 0.85,     # Repulsive signal (right)
        "AWAR": 0.8,      # Avoidance right
        "AWBR": 0.7,
    },
    "disgust": {
        "ASER": 0.9,      # Strong repulsion (right)
        "AWBR": 0.85,
        "ASH": 0.6,       # Touch-chemical avoidance
    },
    "fear": {
        "ASER": 0.75,     # Repulsion right
        "AWAR": 0.85,
        "ASI": 0.5,       # Heightened sensitivity
    },

    # Intensity/Arousal → Central neurons
    "anger": {
        "ASI": 0.9,       # Feeding/intensity drive
        "ASJ": 0.85,      # Aggressive response
        "ASEL": 0.6,
        "ASER": 0.6,
    },
    "surprise": {
        "ASI": 0.8,       # Novel stimulus detection
        "ASJ": 0.7,
        "AWC": 0.6,       # Environmental change
    },
}


def compute_chemosensory_activation(emotions: dict) -> dict:
    """
    Given emotion scores, compute which chemosensory neurons activate and how much.

    Args:
        emotions: dict with keys joy, sadness, fear, etc. and values 0-1

    Returns:
        dict mapping neuron names to activation levels 0-1
    """
    neuron_activation = {}

    for emotion, score in emotions.items():
        if score == 0 or emotion not in EMOTION_TO_NEURONS:
            continue

        # Get neurons activated by this emotion
        neurons = EMOTION_TO_NEURONS[emotion]
        for neuron, base_level in neurons.items():
            # Activation = base level × emotion intensity
            activation = base_level * score
            # Take max if neuron is already activated by another emotion
            neuron_activation[neuron] = max(
                neuron_activation.get(neuron, 0),
                activation
            )

    return neuron_activation


def get_neuron_side(neuron: str) -> str:
    """Return 'L' (left), 'R' (right), or 'C' (center) for a chemosensory neuron."""
    if neuron.endswith("L"):
        return "L"
    elif neuron.endswith("R"):
        return "R"
    else:
        return "C"


def get_neuron_color(neuron: str) -> tuple:
    """Return (hue, saturation, lightness) for a neuron type."""
    # Group neurons by type
    if neuron.startswith("ASE"):
        return (240, 100, 50)  # Blue - basic valence
    elif neuron.startswith("AWA"):
        return (120, 100, 50)  # Green - appetitive
    elif neuron.startswith("AWB"):
        return (150, 100, 50)  # Cyan - approach
    elif neuron.startswith("AWC"):
        return (180, 100, 50)  # Turquoise - CO2
    elif neuron.startswith("ASI"):
        return (60, 100, 50)  # Yellow - intensity
    elif neuron.startswith("ASJ"):
        return (30, 100, 50)  # Orange - feeding
    elif neuron.startswith("ASH"):
        return (0, 100, 50)   # Red - protective
    else:
        return (210, 100, 50)  # Default purple
