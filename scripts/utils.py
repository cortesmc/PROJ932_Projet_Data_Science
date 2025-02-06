def ensure_hashable(entity):
    """
    Convert an entity to a hashable type (e.g., tuple or string).
    
    Args:
        entity: The entity to convert (can be a list, dict, or other types).
    
    Returns:
        hashable: Hashable version of the entity.
    
    Raises:
        ValueError: If the entity cannot be converted to a hashable type.
    """
    if entity is None:
        return "Unknown"  # Placeholder for missing values
    
    if isinstance(entity, list):
        if len(entity) == 1:
            return str(entity[0])  # Convert single-item lists to a string
        return tuple(str(e) for e in entity)  # Convert lists to tuple of strings
    
    if isinstance(entity, dict):
        return tuple((str(k), str(v)) for k, v in entity.items())  # Convert dict to tuple of key-value pairs
    
    if isinstance(entity, (str, int, float, tuple)):
        return str(entity)  # Ensure consistency by converting everything to string
    
    raise ValueError(f"Unhashable entity: {entity} of type {type(entity)}")
