import runpy
import sys
import urllib.error

import pytest
import setuptools
import torch


@pytest.mark.parametrize(
    ("cuda_version", "machine", "platform_tag"),
    [
        ("12.9", "aarch64", "linux_aarch64"),
        ("13.0", "x86_64", "linux_x86_64"),
    ],
)
def test_wheel_url_matches_release_asset_names(
    monkeypatch, cuda_version, machine, platform_tag
):
    monkeypatch.setenv("FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD", "TRUE")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    setup_module = runpy.run_path("setup.py")
    monkeypatch.setattr(setup_module["platform"], "machine", lambda: machine)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu130")
    monkeypatch.setattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True)

    _, wheel_name = setup_module["get_wheel_url"]()

    package_version = setup_module["get_package_version"]()
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    assert wheel_name == (
        f"fast_hadamard_transform-{package_version}"
        f"+cu{cuda_version.split('.')[0]}torch2.11cxx11abiTRUE"
        f"-{python_tag}-{python_tag}-{platform_tag}.whl"
    )


def test_missing_architecture_wheel_falls_back_to_source_build(monkeypatch):
    monkeypatch.setenv("FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD", "TRUE")
    monkeypatch.setenv("FAST_HADAMARD_TRANSFORM_FORCE_BUILD", "FALSE")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    setup_module = runpy.run_path("setup.py")
    monkeypatch.setattr(setup_module["platform"], "machine", lambda: "ppc64le")
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu129")

    def missing_wheel(url, filename):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(setup_module["urllib"].request, "urlretrieve", missing_wheel)
    source_builds = []
    monkeypatch.setattr(
        setup_module["_bdist_wheel"],
        "run",
        lambda command: source_builds.append(command),
    )
    command = setup_module["CachedWheelsCommand"](setuptools.Distribution())

    command.run()

    assert source_builds == [command]
