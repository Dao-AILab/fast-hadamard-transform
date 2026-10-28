import errno
import importlib.util
import shutil
from pathlib import Path

import setuptools


def test_cached_wheel_move_across_filesystems(tmp_path, monkeypatch):
    source_root = Path(__file__).parents[1]
    shutil.copy2(source_root / "README.md", tmp_path / "README.md")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD", "TRUE")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    spec = importlib.util.spec_from_file_location(
        "setup_under_test", source_root / "setup.py"
    )
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)

    wheel_filename = "cached-wheel.whl"
    monkeypatch.setattr(
        setup_module,
        "get_wheel_url",
        lambda: ("https://example.invalid/wheel", wheel_filename),
    )

    def download(_url, filename):
        Path(filename).write_bytes(b"cached wheel contents")

    monkeypatch.setattr(setup_module.urllib.request, "urlretrieve", download)
    command = setup_module.CachedWheelsCommand(setuptools.Distribution())
    command.ensure_finalized()
    command.dist_dir = str(tmp_path / "dist")
    monkeypatch.setattr(command, "get_tag", lambda: ("cp312", "cp312", "linux_x86_64"))

    def cross_device_rename(_source, destination):
        raise OSError(errno.EXDEV, "Invalid cross-device link", str(destination))

    monkeypatch.setattr(setup_module.os, "rename", cross_device_rename)

    command.run()

    [wheel_path] = (tmp_path / "dist").glob("*.whl")
    assert wheel_path.read_bytes() == b"cached wheel contents"
    assert not (tmp_path / wheel_filename).exists()
