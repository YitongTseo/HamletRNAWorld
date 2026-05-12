"""Map word emotions to chemosensory neuron activations.

BIOLOGICAL BACKGROUND:
All chemosensory neurons are located in the AMPHID SENSILLA — bilateral structures
in the worm's nose (head). They come in paired left/right versions that allow the
worm to sense which direction a smell comes from (bilateral chemotaxis).

NEURON PAIRS AND THEIR NATURAL FUNCTIONS:
────────────────────────────────────────

ASE (ASEL/ASER): PRIMARY VALENCE DETECTION — Salt sensing
  - ASEL: Attracted to NaCl (attractive chemicals) — "good smell"
  - ASER: Repelled by high salt (repulsive) — "bad smell"
  - In nature: Navigate toward food sources, away from danger
  - Emotional mapping: Opposite valences (ASEL=approach, ASER=avoid)

AWA (AWAL/AWAR): APPETITIVE ODOR SENSING — Food odors
  - Both L/R: Attracted to volatile food odors (diacetyl, benzaldehyde)
  - In nature: Seek out bacterial food sources
  - Emotional mapping: Positive/approach emotions activate both sides

AWB (AWBL/AWBR): BEHAVIORAL APPROACH — Complex odor sensing
  - Both L/R: Involved in odor-evoked approach behavior
  - In nature: Navigate toward favorable environments
  - Emotional mapping: Trust, anticipation

AWC (AWCL/AWCR): CO2 DETECTION — Environmental awareness
  - Both L/R: Sense CO2 levels (high = aversive, indicates low oxygen)
  - In nature: Avoid hypoxic/crowded environments
  - Emotional mapping: Safety, surprise (environmental change)

ASI (ASIL/ASIR): POLYMODAL FEEDING DRIVE — Multisensory integration
  - Both L/R: Feeding state-dependent. Active when hungry.
  - In nature: Modulate feeding and general arousal
  - Emotional mapping: Anger, surprise (intensity/urgency)

ASJ (ASJL/ASJR): POLYMODAL NOVELTY/FEEDING — Taste + novelty
  - Both L/R: Respond to novel stimuli AND feeding state
  - In nature: Detect food (taste) and novelty simultaneously
  - Emotional mapping: Anger, surprise (high arousal)

ASH (ASHL/ASHR): PROTECTIVE NOCICEPTOR — Pain/avoidance
  - Both L/R: Respond to aversive touch and osmotic stress
  - In nature: Protective reflexes, avoidance learning
  - Emotional mapping: Disgust, fear (strong avoidance)

ASG (ASGL/ASGR): POLYMODAL TOUCH/CHEMICAL — Integrated sensing
  - Both L/R: Respond to touch AND certain chemicals
  - In nature: Coordinate touch responses with chemical cues
  - Emotional mapping: Disgust (protective response)

ASK (ASKL/ASKR): POLYMODAL PROTECTIVE — Similar to ASH
  - Both L/R: Nociceptive, aversive stimulus detection
  - In nature: Protective reflexes, defensive behavior
  - Emotional mapping: Fear, disgust

ADF (ADFL/ADFR): AMPHID SENSILLA — Food-related odors
  - Both L/R: Similar to AWA, attracted to food odors
  - In nature: Seek out bacterial food sources
  - Emotional mapping: Joy, trust (approach)

ADL (ADLL/ADLR): AMPHID SENSILLA — Polymodal sensory
  - Both L/R: Sense multiple modalities (touch, chemical, osmotic)
  - In nature: Integrated environmental sensing
  - Emotional mapping: Mixed emotions, novelty

EMOTION MAPPING STRATEGY:
────────────────────────
- Positive emotions (joy, trust) → Approach neurons (ASE-left, AWA, AWB, ADF)
- Negative emotions (fear, disgust) → Avoidance neurons (ASE-right, ASH, ASK)
- High-arousal emotions (anger, surprise) → Feeding/intensity (ASI, ASJ)
- Each emotion maps to BOTH L and R for bilateral directional sensing
"""

from __future__ import annotations

# Mapping emotions to chemosensory neuron activation patterns
# Each emotion maps to both L and R versions so the worm can sense direction
EMOTION_TO_NEURONS = {
    # POSITIVE/APPROACH EMOTIONS — Activate toward-response neurons
    "joy": {
        # ASE: Primary valence detection — joy signals good things (food, safety)
        "ASEL": 0.9,      # Salt/attractive — bilateral coverage
        "ASER": 0.9,
        # AWA: Appetitive odor — strong food-seeking response
        "AWAL": 0.85,
        "AWAR": 0.85,
        # AWB: Approach behavior — move toward the stimulus
        "AWBL": 0.7,
        "AWBR": 0.7,
    },
    "trust": {
        # ASE: Moderate attractive signal
        "ASEL": 0.75,
        "ASER": 0.75,
        # AWA: Appetitive seeking
        "AWAL": 0.8,
        "AWAR": 0.8,
        # AWC: CO2 safety signal — trust = safe environment
        "AWCL": 0.5,
        "AWCR": 0.5,
    },
    "anticipation": {
        # ASE: Moderate valence signal
        "ASEL": 0.6,
        "ASER": 0.6,
        # AWA: Active seeking behavior
        "AWAL": 0.7,
        "AWAR": 0.7,
        # AWC: Environmental monitoring — anticipate change
        "AWCL": 0.5,
        "AWCR": 0.5,
    },

    # NEGATIVE/AVOIDANCE EMOTIONS — Activate away-response neurons
    "sadness": {
        # ASE: Weak valence signal (withdrawal)
        "ASEL": 0.7,
        "ASER": 0.7,
        # AWA: Reduced appetitive drive (low motivation)
        "AWAL": 0.65,
        "AWAR": 0.65,
        # AWB: Behavioral withdrawal/approach reduction
        "AWBL": 0.6,
        "AWBR": 0.6,
    },
    "disgust": {
        # ASE: Weak positive signal (want to escape)
        "ASEL": 0.5,
        "ASER": 0.5,
        # AWB: Strong avoidance response
        "AWBL": 0.85,
        "AWBR": 0.85,
        # ASH: Protective nociceptor — strong rejection/avoidance
        "ASHL": 0.6,
        "ASHR": 0.6,
    },
    "fear": {
        # ASE: Weak valence (threat detected)
        "ASEL": 0.6,
        "ASER": 0.6,
        # AWA: Reduced food seeking (fear inhibits appetite)
        "AWAL": 0.5,
        "AWAR": 0.5,
        # ASI: Heightened sensitivity / arousal in fear state
        "ASIL": 0.5,
        "ASIR": 0.5,
    },

    # HIGH-AROUSAL EMOTIONS — Activate intensity/feeding drive neurons
    "anger": {
        # ASI: Feeding drive / aggression (anger = arousal + approach)
        "ASIL": 0.9,      # Bilateral feeding/intensity
        "ASIR": 0.9,
        # ASJ: Novelty + feeding response (respond aggressively)
        "ASJL": 0.85,
        "ASJR": 0.85,
        # ASE: Mild valence signal (focused on target)
        "ASEL": 0.5,
        "ASER": 0.5,
    },
    "surprise": {
        # ASI: Novel stimulus detection — high sensitivity
        "ASIL": 0.8,      # Bilateral novelty/arousal
        "ASIR": 0.8,
        # ASJ: Feeding + novelty response
        "ASJL": 0.7,
        "ASJR": 0.7,
        # AWC: CO2/environmental change detection
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
    elif neuron.startswith("ASK"):
        return (15, 100, 50)  # Red-orange - protective
    elif neuron.startswith("ASG"):
        return (45, 100, 50)  # Orange-red - defensive
    elif neuron.startswith("ADF"):
        return (100, 100, 50)  # Yellow-green - food
    elif neuron.startswith("ADL"):
        return (210, 100, 50)  # Purple - general
    else:
        return (210, 100, 50)  # Default purple


# DETAILED NEURON REFERENCE
NEURON_REFERENCE = {
    "ASEL": {
        "name": "Amphid Sensilla Left",
        "type": "ASE",
        "side": "L",
        "function": "Salt/ion detection - attracted to NaCl (food sodium). Primary valence sensor for attractive cues.",
        "emotions": "joy, trust, anticipation (approach)",
        "nature": "In real C. elegans: Navigate toward food sources via salt gradients",
    },
    "ASER": {
        "name": "Amphid Sensilla Right",
        "type": "ASE",
        "side": "R",
        "function": "Salt/ion detection - repelled by high osmolarity (danger). Primary valence sensor for repulsive cues.",
        "emotions": "joy, trust, anticipation, disgust, fear (avoid)",
        "nature": "In real C. elegans: Avoid hyperosmotic or harmful environments",
    },
    "AWAL": {
        "name": "Amphid Sensilla Left",
        "type": "AWA",
        "side": "L",
        "function": "Volatile odor sensing - attracted to food odors (diacetyl, benzaldehyde). Appetitive approach.",
        "emotions": "joy, trust, anticipation (seek food)",
        "nature": "In real C. elegans: Find bacterial food at distance",
    },
    "AWAR": {
        "name": "Amphid Sensilla Right",
        "type": "AWA",
        "side": "R",
        "function": "Volatile odor sensing - same as AWAL but on right side. Bilateral food seeking.",
        "emotions": "joy, trust, anticipation (seek food)",
        "nature": "In real C. elegans: Bilateral comparison for odor gradient tracking",
    },
    "AWBL": {
        "name": "Amphid Sensilla Left",
        "type": "AWB",
        "side": "L",
        "function": "Volatile odor / behavioral - supports approach behavior. Complex integration.",
        "emotions": "joy (approach), sadness (reduce approach)",
        "nature": "In real C. elegans: Modulate approach/avoidance based on odor context",
    },
    "AWBR": {
        "name": "Amphid Sensilla Right",
        "type": "AWB",
        "side": "R",
        "function": "Volatile odor / behavioral - bilateral support for approach/avoidance.",
        "emotions": "joy (approach), sadness (reduce approach)",
        "nature": "In real C. elegans: Bilateral behavioral control",
    },
    "AWCL": {
        "name": "Amphid Sensilla Left",
        "type": "AWC",
        "side": "L",
        "function": "CO2 sensing - detect high CO2 as aversive (indicates hypoxia/crowding).",
        "emotions": "trust (safety), anticipation (monitor environment), surprise (change)",
        "nature": "In real C. elegans: Avoid low-oxygen or overcrowded areas",
    },
    "AWCR": {
        "name": "Amphid Sensilla Right",
        "type": "AWC",
        "side": "R",
        "function": "CO2 sensing - bilateral environmental monitoring.",
        "emotions": "trust (safety), anticipation (monitor environment), surprise (change)",
        "nature": "In real C. elegans: Bilateral CO2 gradient detection",
    },
    "ASIL": {
        "name": "Amphid Sensilla Left",
        "type": "ASI",
        "side": "L",
        "function": "Polymodal sensory - feeding-state dependent. Integrates hunger + environmental cues.",
        "emotions": "anger, surprise (high arousal/intensity)",
        "nature": "In real C. elegans: Modulate all behavior based on metabolic state",
    },
    "ASIR": {
        "name": "Amphid Sensilla Right",
        "type": "ASI",
        "side": "R",
        "function": "Polymodal sensory - bilateral feeding state sensing.",
        "emotions": "anger, surprise (high arousal/intensity)",
        "nature": "In real C. elegans: Bilateral arousal regulation",
    },
    "ASJL": {
        "name": "Amphid Sensilla Left",
        "type": "ASJ",
        "side": "L",
        "function": "Polymodal - taste + novelty detection. Responds to novel and familiar stimuli differently based on feeding state.",
        "emotions": "anger, surprise (novel/intense stimulus)",
        "nature": "In real C. elegans: Detect food AND learn/remember odors",
    },
    "ASJR": {
        "name": "Amphid Sensilla Right",
        "type": "ASJ",
        "side": "R",
        "function": "Polymodal - bilateral taste/novelty sensing.",
        "emotions": "anger, surprise (novel/intense stimulus)",
        "nature": "In real C. elegans: Bilateral novelty/learning coordination",
    },
    "ASHL": {
        "name": "Amphid Sensilla Left",
        "type": "ASH",
        "side": "L",
        "function": "Nociceptor (pain sensor) - responds to touch + noxious chemicals (osmotic stress). Protective reflex.",
        "emotions": "disgust, fear (strong avoidance)",
        "nature": "In real C. elegans: Protective withdrawal and avoidance learning",
    },
    "ASHR": {
        "name": "Amphid Sensilla Right",
        "type": "ASH",
        "side": "R",
        "function": "Nociceptor - bilateral protective response.",
        "emotions": "disgust, fear (strong avoidance)",
        "nature": "In real C. elegans: Bilateral protective coordination",
    },
}
