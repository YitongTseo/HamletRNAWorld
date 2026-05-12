"""Map word emotions to chemosensory neuron activations.

In C. elegans, chemosensory neurons come in bilateral pairs:
- Each emotion maps to BOTH left (L) and right (R) versions of sensory pairs
- This allows the worm to sense which direction a smell comes from
- A smell to the left will activate L neurons more strongly than R
- A smell to the right will activate R neurons more strongly than L

Neuron pairs and their roles:
- ASE (ASEL/ASER): Primary valence detection
- AWA (AWAL/AWAR): Appetitive approach
- AWB (AWBL/AWBR): Behavioral approach/avoidance
- AWC (AWCL/AWCR): Safety/CO2
- ASI (ASIL/ASIR): Feeding drive/intensity
- ASJ (ASJL/ASJR): Feeding/novelty
- ASH (ASHL/ASHR): Protective/avoidance
"""

from __future__ import annotations

# Mapping emotions to chemosensory neuron activation patterns
# Each emotion maps to both L and R versions so the worm can sense direction
EMOTION_TO_NEURONS = {
    # Positive/Approach emotions
    "joy": {
        "ASEL": 0.9,      # Attractive left
        "ASER": 0.9,      # Attractive right (bilateral coverage)
        "AWAL": 0.85,
        "AWAR": 0.85,
        "AWBL": 0.7,
        "AWBR": 0.7,
    },
    "trust": {
        "ASEL": 0.75,
        "ASER": 0.75,
        "AWAL": 0.8,
        "AWAR": 0.8,
        "AWCL": 0.5,
        "AWCR": 0.5,
    },
    "anticipation": {
        "ASEL": 0.6,      # Both sides seek forward
        "ASER": 0.6,
        "AWAL": 0.7,
        "AWAR": 0.7,
        "AWCL": 0.5,
        "AWCR": 0.5,
    },

    # Negative/Avoid emotions
    "sadness": {
        "ASEL": 0.7,
        "ASER": 0.7,
        "AWAL": 0.65,
        "AWAR": 0.65,
        "AWBL": 0.6,
        "AWBR": 0.6,
    },
    "disgust": {
        "ASEL": 0.5,
        "ASER": 0.5,
        "AWBL": 0.85,
        "AWBR": 0.85,
        "ASHL": 0.6,
        "ASHR": 0.6,
    },
    "fear": {
        "ASEL": 0.6,
        "ASER": 0.6,
        "AWAL": 0.5,
        "AWAR": 0.5,
        "ASIL": 0.5,
        "ASIR": 0.5,
    },

    # Intensity/Arousal → Bilateral symmetric
    "anger": {
        "ASIL": 0.9,      # Bilateral feeding drive
        "ASIR": 0.9,
        "ASJL": 0.85,
        "ASJR": 0.85,
        "ASEL": 0.5,
        "ASER": 0.5,
    },
    "surprise": {
        "ASIL": 0.8,      # Bilateral novelty detection
        "ASIR": 0.8,
        "ASJL": 0.7,
        "ASJR": 0.7,
        "AWCL": 0.6,
        "AWCR": 0.6,
    },
}


def compute_chemosensory_activation(emotions: dict, direction_factor: float = 0.5) -> dict:
    """
    Given emotion scores, compute which chemosensory neurons activate and how much.

    Args:
        emotions: dict with keys joy, sadness, fear, etc. and values 0-1
        direction_factor: 0 = food to right (boost R), 1 = food to left (boost L), 0.5 = straight

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

            # Apply directional weighting: L neurons stronger when food is left (direction_factor > 0.5)
            if neuron.endswith("L"):
                activation *= direction_factor  # 0.5-1.0 range
            elif neuron.endswith("R"):
                activation *= (1.0 - direction_factor)  # 1.0-0.5 range

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
