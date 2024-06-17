def _validate(condition: bool, message: str) -> None:
    if condition is False:
        raise ValueError(message)
