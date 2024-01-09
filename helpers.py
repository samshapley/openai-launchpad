def log(logging_enabled, message, color=None):
    """
    Logs a message to the console if logging is enabled, with an optional color.

    Args:
        message (str): The message to log.
        color (str, optional): The color name for the message. Defaults to None.
    """
    # ANSI color codes
    COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "purple": "\033[95m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "end": "\033[0m",  # Reset to default color
    }
    if logging_enabled:
        if color in COLORS:
            print(COLORS[color] + message + COLORS["end"])
        else:
            print(message)
        print('---------------------------------------------')

def to_dict(obj):
    """
    Recursively convert an object into a dictionary.
    Handles nested objects and lists of objects.
    """
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(element) for element in obj]
    elif hasattr(obj, '__dict__'):
        return to_dict(vars(obj))
    else:
        return obj