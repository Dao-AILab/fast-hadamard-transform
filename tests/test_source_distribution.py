import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def test_source_distribution_contains_cuda_sources(tmp_path):
    source_root = Path(__file__).parents[1]
    project = tmp_path / "project"
    project.mkdir()

    for filename in ("AUTHORS", "LICENSE", "README.md", "setup.py"):
        shutil.copy2(source_root / filename, project / filename)
    for directory in ("csrc", "fast_hadamard_transform"):
        shutil.copytree(source_root / directory, project / directory)
    manifest = source_root / "MANIFEST.in"
    if manifest.exists():
        shutil.copy2(manifest, project / "MANIFEST.in")

    distribution_dir = tmp_path / "dist"
    environment = os.environ.copy()
    environment["FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD"] = "TRUE"
    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "sdist",
            "--dist-dir",
            str(distribution_dir),
        ],
        cwd=project,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    archives = list(distribution_dir.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        archived_files = set(archive.getnames())

    package_root = archives[0].name[: -len(".tar.gz")]
    required_sources = {
        path.relative_to(project).as_posix()
        for path in (project / "csrc").rglob("*")
        if path.suffix in {".cpp", ".cu", ".h"}
    }
    missing_sources = {
        f"{package_root}/{path}" for path in required_sources
    } - archived_files

    assert not missing_sources, f"Source archive is missing: {sorted(missing_sources)}"
