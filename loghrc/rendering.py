"""Compatibility shim. Prefer loghrc.viz.rendering."""

from loghrc.viz.rendering import export_video, main, render_grid  # noqa: F401

if __name__ == "__main__":
    main()
