#!/usr/bin/env python3


def get_activity_name_to_intensity_mappings(version="v0.4") -> str:
    """
    Maps an activity name to an intensity level.
    """
    # Intensity-based grouping into four categories
    activity_groups = {
        "Sedentary": [
            # Typically near-rest or unclassified:
            "other", 
            "sonstige",
            "autre", 
            "cooldown", 
            "preparationandrecovery", 
            "atemübung", 
            "fitnessgaming", 
            "chronomètre", 
            "incident detected", 
            "exercice de respiration",
        ],
        "Light": [
            # Light-intensity / recovery, often <3 METs:
            "walking", 
            "gehen", 
            "marche à pied", 
            "spazieren gehen", 
            "camminata", 
            "marche",
            "yoga", 
            "pilates", 
            "flexibility", 
            "mindandbody", 
            "taichi",
            "equestriansports",
            # ADDED:
            "golf",  # often low intensity
        ],
        "Moderate": [
            # Moderate-intensity activities (~3–6 METs):
            
            # Hiking
            "hiking", 
            "wandern", 
            "randonnée", 
            "bergsteigen", 
            "alpinisme",
            
            # Running (generic pace)
            "running", 
            "laufen", 
            "course à pied", 
            "course",
            "trailrunning", 
            "treadmill_running", 
            "treadmill running", 
            "indoor_running",
            "corsa", 
            "trail running", 
            "trail_running", 
            "course à pied urbaine", 
            "laufen auf der bahn", 
            "course à pied sur tapis roulant", 
            "trail",
            "indoor-laufen",    # treadmill-like
            "laufbandtraining", # treadmill-like
            
            # Cycling (generic)
            "cycling", 
            "radfahren", 
            "indoor cycling", 
            "virtuelles radfahren", 
            "cyclisme",
            "cyclisme sur route", 
            "mountainbiken", 
            "mountain biking", 
            "virtual cycling",
            "vélo d'intérieur", 
            "vtt", 
            "rennradfahren", 
            "e-mountainbike-fahren", 
            "gravel bike", 
            "road cycling", 
            "ebiking", 
            "gravel/offroad-radfahren",
            "indoor_cycling",   # alternate spelling
            
            # Swimming
            "swimming", 
            "pool swimming", 
            "lap_swimming", 
            "schwimmbadschwimmen",
            "nuoto in piscina", 
            "en piscine", 
            "open water swimming", 
            "open_water_swimming",
            "waterfitness", 
            "schwimmen",
            
            # Ball sports, skating, dance
            "basketball", 
            "tennis", 
            "volleyball", 
            "badminton", 
            "tabletennis", 
            "squash",
            "skatingsports", 
            "skating", 
            "inlineskaten", 
            "dance", 
            "stairs",
        ],
        "High": [
            # High/Vigorous intensity (>6 METs)
            "highintensityintervaltraining", 
            "crosstraining", 
            "mixedcardio", 
            "elliptical", 
            "cardio", 
            "mixedmetaboliccardiotraining", 
            "indoor_cardio",
            "jumprope", 
            "stairclimbing", 
            "stepper", 
            "hiit", 
            "cardiodance",
            "boxing", 
            "kickboxing", 
            "gymnastics",
            
            # Snow sports
            "snowsports", 
            "downhillskiing", 
            "crosscountryskiing", 
            "skifahren/snowboarden",
            "klassischer langlauf", 
            "resort skiing/snowboarding", 
            "langlauf freistil",
            "ski de randonnée nordique/surf des neiges", 
            "backcountry skiing/snowboarding",
            "ski de fond skating", 
            "ski sur piste/surf des neiges", 
            "snowboarding",
            "cross country classic skiing", 
            "snowshoeing", 
            "cross country skate skiing",
            "ski-/snowboardtour",
            
            # Rowing / water / paddle
            "rowing", 
            "rudern", 
            "rudermaschine", 
            "indoor rowing",
            "sailing", 
            "paddlesports", 
            "stand up paddleboarding", 
            "stand-up-paddle-boarding",
            "watersports", 
            "surfingsports", 
            "surfen", 
            "windsurfing", 
            "kayaking",
            
            "multi_sport", 
            "multisport",
            
            "climbing",       # can be quite vigorous
            "rock climbing",  # same reasoning
            "bouldern",       # bouldering is typically high-intensity
        ],
        "Strength": [
            # Resistance / strength training
            "traditionalstrengthtraining", 
            "functionalstrengthtraining", 
            "coretraining", 
            "krafttraining", 
            "musculation", 
            "strength training",
        ],
    }




    # Create an inverse mapping: for each activity, assign its group name.
    activity_to_group = {}
    for group_name in activity_groups.keys():
        for activity in activity_groups[group_name]:
            activity_to_group[activity] = group_name

    return activity_to_group


def get_activity_name_to_met_score_mapping(): 
    return {
        "laufen": 8.0,  # Running (general)
        "sonstige": 4.0,  # Other/unspecified moderate activity
        "krafttraining": 6.0,  # Strength training
        "walking": 3.5,  # Walking (3.0–4.0 range for moderate pace)
        "indoor_cardio": 6.0,  # Generic indoor cardio
        "running": 8.0,  # Running (general)
        "hiking": 6.0,  # Hiking
        "other": 4.0,  # Other/unspecified moderate activity
        "virtuelles radfahren": 7.0,  # Virtual cycling (moderate)
        "skifahren/snowboarden": 6.0,  # Downhill skiing/snowboarding (moderate)
        "wandern": 6.0,  # Hiking (German)
        "rowing": 8.0,  # Rowing (moderate-vigorous)
        "cycling": 7.5,  # Cycling (moderate, ~12–13.9 mph)
        "traditionalstrengthtraining": 6.0,
        "marche à pied": 3.5,  # Walking (French)
        "yoga": 3.0,
        "course": 8.0,  # Running (French “course à pied”)
        "cardio": 6.0,  # Generic cardio
        "musculation": 6.0,  # Strength training (French)
        "dance": 5.5,
        "coretraining": 4.0,
        "paddlesports": 5.0,  # General paddling (kayak/canoe moderate)
        "treadmill_running": 8.0,
        "downhillskiing": 6.0,
        "stairs": 8.0,  # Stair climbing
        "squash": 7.3,  # Can range 7–12, average ~7.3
        "swimming": 6.0,  # Moderate effort
        "mixedcardio": 6.0,
        "gehen": 3.5,  # Walking (German)
        "radfahren": 7.5,  # Cycling (German) moderate
        "highintensityintervaltraining": 8.0,  # HIIT can vary widely; ~8+ for moderate-vigorous
        "autre": 4.0,  # Other (French)
        "elliptical": 5.0,  # Moderate use of elliptical trainer
        "mountain biking": 8.5,
        "resort skiing/snowboarding": 6.0,
        "gymnastics": 3.8,
        "mixedmetaboliccardiotraining": 8.0,
        "skatingsports": 7.0,  # Ice/inline skating (moderate)
        "crosstraining": 8.0,
        "indoor_running": 8.0,
        "strength training": 6.0,
        "indoor cycling": 6.8,  # Spin class/indoor bike (moderate)
        "stairclimbing": 8.0,
        "mindandbody": 2.5,  # Light yoga/meditation
        "volleyball": 4.0,  # Recreational
        "climbing": 8.0,  # Rock climbing (moderate)
        "snowsports": 6.0,
        "crosscountryskiing": 8.0,  # Moderate cross-country
        "tennis": 7.3,  # Singles
        "kickboxing": 7.5,
        "surfingsports": 5.0,  # Surfing
        "sailing": 3.0,
        "pilates": 3.5,
        "treadmill running": 8.0,
        "lap_swimming": 7.0,  # Continuous lap swimming (moderate-vigorous)
        "trail_running": 9.0,  # Usually higher than road running
        "stand-up-paddle-boarding": 6.0,
        "exercice de respiration": 1.3,  # Breathing exercise
        "ski sur piste/surf des neiges": 6.0,  # Downhill ski/snowboard (French)
        "vélo d'intérieur": 6.8,  # Indoor cycling (French)
        "ski de randonnée nordique/surf des neiges": 9.0,  # Backcountry touring
        "ski de fond skating": 9.5,  # Cross-country skate skiing
        "randonnée": 6.0,  # Hiking (French)
        "atemübung": 1.3,  # Breathing exercise (German)
        "watersports": 5.0,  # Generic water sports
        "basketball": 6.5,  # Recreational game
        "badminton": 5.5,
        "functionalstrengthtraining": 6.0,
        "stand up paddleboarding": 6.0,
        "taichi": 3.0,
        "inlineskaten": 7.0,  # Inline skating (German)
        "preparationandrecovery": 3.0,  # Light warm-up/cool-down
        "indoor_cycling": 6.8,  # Duplicate form
        "multi_sport": 6.0,  # Mixed moderate activities
        "open_water_swimming": 9.0,  # Usually more vigorous
        "schwimmbadschwimmen": 6.0,  # Pool swimming (German)
        "langlauf freistil": 9.5,  # XC skiing freestyle (German)
        "rennradfahren": 10.0,  # Road cycling (racing pace)
        "spazieren gehen": 3.5,  # Walking (German)
        "mountainbiken": 8.5,  # Mountain biking (German)
        "cardiodance": 6.0,
        "laufbandtraining": 8.0,  # Treadmill training (German)
        "marche": 3.5,  # Walking (French)
        "cyclisme": 7.5,  # Cycling (French)
        "open water swimming": 9.0,
        "chronomètre": 4.0,  # “Stopwatch” – treating as “other” moderate
        "vtt": 8.5,  # Mountain biking (French)
        "course à pied": 8.0,  # Running (French)
        "incident detected": 0.0,  # Not a physical activity
        "flexibility": 2.3,  # Light stretching
        "cooldown": 2.3,
        "tabletennis": 4.0,
        "cyclisme sur route": 10.0,  # Road cycling (French) at higher intensity
        "gravel bike": 9.0,  # Gravel/off-road cycling
        "klassischer langlauf": 9.0,  # Classic cross-country skiing (German)
        "trailrunning": 9.0,  # Trail running
        "laufen auf der bahn": 8.0,  # Track running
        "multisport": 6.0,  # Mixed moderate
        "bouldern": 8.0,  # Bouldering
        "course à pied urbaine": 8.0,  # Urban running
        "trail": 9.0,  # Trail running
        "golf": 4.8,  # Walking, carrying clubs
        "corsa": 8.0,  # Running (Italian)
        "camminata": 3.5,  # Walking (Italian)
        "nuoto in piscina": 6.0,  # Pool swimming (Italian)
        "jumprope": 10.0,  # Skipping rope (moderate-vigorous)
        "backcountry skiing/snowboarding": 9.0,
        "cross country skate skiing": 9.5,
        "trail running": 9.0,
        "snowshoeing": 8.0,
        "snowboarding": 5.3,  # Downhill moderate
        "boxing": 7.5,
        "fitnessgaming": 5.0,  # e.g., active video games
        "rudermaschine": 7.0,  # Rowing machine (German)
        "stepper": 8.0,  # Stair/step machine
        "bergsteigen": 9.0,  # Mountaineering (German)
        "en piscine": 6.0,  # Swimming in pool (French)
        "alpinisme": 9.0,  # Mountaineering (French)
        "indoor-laufen": 8.0,  # Indoor running (German)
        "schwimmen": 6.0,  # Swimming (German)
        "skating": 7.0,  # Ice/inline skating
        "windsurfing": 6.0,
        "pool swimming": 6.0,
        "ebiking": 4.0,  # E-bike moderate
        "kayaking": 5.0,  # Moderate effort
        "cross country classic skiing": 9.0,
        "road cycling": 10.0,  # Vigorous road cycling
        "waterfitness": 3.0,  # Water aerobics
        "ski-/snowboardtour": 9.0,  # Ski touring
        "indoor rowing": 7.0,
        "surfen": 5.0,  # Surfing (German)
        "course à pied sur tapis roulant": 8.0,  # Treadmill running (French)
        "rock climbing": 8.0,
        "virtual cycling": 7.0,
        "equestriansports": 4.0,  # Horseback riding
        "gravel/offroad-radfahren": 9.0,  # Gravel/off-road cycling (German)
        "e-mountainbike-fahren": 5.5,  # E-MTB moderate
        "rudern": 7.0,  # Rowing (German)
        "hiit": 8.5  # High Intensity Interval Training
    }
        