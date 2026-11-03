# Loaded by every Python process when scripts/fs_site is on PYTHONPATH: installs the Moshi frame-sync gate hook.
try:
    from moshi_neuron.serving.framesync import install_import_hook
    install_import_hook()
except Exception as _e:  # noqa: BLE001
    import sys
    sys.stderr.write(f"[MoshiFS] hook not installed: {_e}\n")
