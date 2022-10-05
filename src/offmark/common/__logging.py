import logging

logger = logging.getLogger(__name__)


def trace(module_logger):
    def decorator(fn):
        def inner(*args, **kwargs):
            module_logger.debug(f'Entering {fn.__name__}()')  # print args?
            result = fn(*args, **kwargs)
            # Exiting
            return result

        return inner

    return decorator
