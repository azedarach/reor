from .pyreor_ext import backends

def _check_backend(backend, whom):
    if backend not in backends():
        raise RuntimeError(
            "Unsupported backend '%s' passed to %s. "
            "The backend must be one of: %s" %
            (backend, whom, backends()))
