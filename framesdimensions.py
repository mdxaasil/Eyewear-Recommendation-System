frame_dimensions = {
    'Cat-eye': {
        'Frame Width': (130, 140),
        'Lens Height': (40, 55),
        'Bridge Size': (15, 20),
        'Temple Length': (135, 150)
    },
    'Rimless': {
        'Frame Width': None,  # Varies significantly
        'Lens Height': None,  # Varies
        'Bridge Size': (15, 20),
        'Temple Length': None  # Varies
    },
    'Aviator': {
        'Frame Width': (130, 150),
        'Lens Height': (50, 60),
        'Bridge Size': (15, 20),
        'Temple Length': (130, 150)
    },
    'Light-Colored': {
        'Frame Width': None,  # Varies
        'Lens Height': None,  # Varies
        'Bridge Size': None,  # Varies
        'Temple Length': None  # Varies
    },
    'Round': {
        'Frame Width': (120, 140),
        'Lens Height': (40, 55),
        'Bridge Size': (15, 20),
        'Temple Length': (130, 150)
    },
    'Oval': {
        'Frame Width': (120, 140),
        'Lens Height': (35, 50),
        'Bridge Size': (15, 20),
        'Temple Length': (130, 150)
    },
    'Wayfarer': {
        'Frame Width': (130, 150),
        'Lens Height': (40, 55),
        'Bridge Size': (15, 20),
        'Temple Length': (130, 150)
    },
    'Oversized': {
        'Frame Width': (140, 160),
        'Lens Height': (50, 70),
        'Bridge Size': (15, 25),
        'Temple Length': (140, 160)
    },
    'Geometric': {
        'Frame Width': (130, 150),
        'Lens Height': (40, 60),
        'Bridge Size': (15, 20),
        'Temple Length': (130, 150)
    },
    'Square': {
        'Frame Width': (130, 150),
        'Lens Height': (40, 60),
        'Bridge Size': (15, 25),
        'Temple Length': (130, 150)
    },
    'Rectangular': {
        'Frame Width': (130, 150),
        'Lens Height': (30, 50),
        'Bridge Size': (15, 20),
        'Temple Length': (130, 150)
    },
    'Angular': {
        'Frame Width': (130, 150),
        'Lens Height': (40, 60),
        'Bridge Size': (15, 25),
        'Temple Length': (130, 150)
    }
}

face_frame_suitability = {
    'Heart': {
        'Dimensions': {
            'Face Width': None,  # Replace with actual values from your model
            'Face Length': None,  # Replace with actual values from your model
        },
        'Suitable Frames': ['Cat-eye', 'Rimless', 'Aviator', 'Light-Colored']
    },
    'Square': {
        'Dimensions': {
            'Face Width': None,  # Replace with actual values from your model
            'Face Length': None,  # Replace with actual values from your model
        },
        'Suitable Frames': ['Round', 'Oval', 'Wayfarer', 'Rimless', 'Light-Colored']
    },
    'Oval': {
        'Dimensions': {
            'Face Width': None,  # Replace with actual values from your model
            'Face Length': None,  # Replace with actual values from your model
        },
        'Suitable Frames': ['Cat-eye', 'Round', 'Oval', 'Wayfarer', 'Rimless']
    },
    'Oblong': {
        'Dimensions': {
            'Face Width': None,  # Replace with actual values from your model
            'Face Length': None,  # Replace with actual values from your model
        },
        'Suitable Frames': ['Oversized', 'Geometric', 'Wayfarer', 'Square']
    },
    'Round': {
        'Dimensions': {
            'Face Width': None,  # Replace with actual values from your model
            'Face Length': None,  # Replace with actual values from your model
        },
        'Suitable Frames': ['Rectangular', 'Angular', 'Wayfarer', 'Square']
    },
}

# Replace 'None' with the actual values predicted by your model for face dimensions.
# You can also extend this structure to include more face shapes and their dimensions as needed.
