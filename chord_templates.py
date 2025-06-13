import numpy as np

# Each template is a PCP vector: [C, C#, D, D#, E, F, F#, G, G#, A,A#, B]
# Based on normalized pitch class presence in ideal chords
chord_templates = np.array([
    # C Major: C E G
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    # C# Major
    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    # D Major: D F# A
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    # D# Major
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    # E Major: E G# B
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    # F Major: F A C
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    # F# Major
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    # G Major: G B D
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    # G# Major
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    # A Major: A C# E
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    # A# Major
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    # B Major: B D# F#
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
])

# Normalize the templates
chord_templates = chord_templates / np.linalg.norm(chord_templates, axis=1, keepdims=True)

# Chord names
chord_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
