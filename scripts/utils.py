def ensure_hashable(entity):
    """
    Convert an entity to a hashable type (e.g., tuple).
    
    Args:
        entity: Entity to convert (can be a list, dict, or other types).
    
    Returns:
        hashable: Hashable version of the entity.
    
    Raises:
        ValueError: If the entity cannot be converted to a hashable type.
    """
    if isinstance(entity, list):
        return tuple(entity)
    elif isinstance(entity, dict):
        return tuple(entity.items())
    elif isinstance(entity, (str, int, float, tuple)):
        return entity
    else:
        raise ValueError(f"Unhashable entity: {entity}")