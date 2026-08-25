"""Compatibility shim. Prefer loghrc.core.validate."""

from loghrc.core.validate import *  # noqa: F401,F403


if __name__ == "__main__":
    from loghrc.core.validate import main

    main()
