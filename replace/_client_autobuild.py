import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

client_dir = Path(__file__).absolute().parent / "client"
build_dir = client_dir / "build"
_SRC_HASH_MARKER = build_dir / ".viser_src_hash"


def _check_viser_dev_running() -> bool:
    """Returns True if the viewer client has been launched via `npm run dev`."""
    try:
        import psutil
    except ImportError:
        # If psutil is not installed, we can't check for dev server.
        # This is fine for normal usage - only needed for development.
        return False

    for process in psutil.process_iter():
        try:
            # Check if the process is running from the correct viser client directory
            # and is actually a vite dev server (not just any vite command)
            cwd = Path(process.cwd()).resolve()
            expected_cwd = client_dir.resolve()

            if cwd == expected_cwd:
                cmdline = process.cmdline()
                # Check for vite with --host flag (which is our dev command)
                # Make sure it's not a build command
                has_vite = any("vite" in part for part in cmdline)
                has_host = any("--host" in part for part in cmdline)
                not_build = not any("build" in part for part in cmdline)

                if has_vite and has_host and not_build:
                    return True
        except (psutil.AccessDenied, psutil.ZombieProcess, psutil.NoSuchProcess):
            pass
    return False


def ensure_client_is_built() -> None:
    """Ensure that the client is built or already running."""

    if not (client_dir / "src").exists():
        # Can't build client.
        assert (build_dir / "index.html").exists(), (
            "Something went wrong! At least one of the client source or build"
            " directories should be present."
        )
        return

    # Do we need to re-trigger a build?
    build = False
    dev_running = _check_viser_dev_running()
    if dev_running:
        # Don't run build if dev server is already running.
        import rich

        rich.print(
            "[bold](viser)[/bold] The Viser viewer looks like it has been launched via"
            " `npm run dev`. Skipping build check..."
        )
        build = False
    elif not (build_dir / "index.html").exists():
        import rich

        rich.print("[bold](viser)[/bold] No client build found. Building now...")
        build = True
    else:
        src_latest_mtime, _ = _modified_time_recursive(client_dir / "src")
        build_latest_mtime, _ = _modified_time_recursive(build_dir)

        if (
            # We should be at least 10 seconds newer than the last build.
            # This buffer is important when we install from pip, and the src/ +
            # build/ directories have very similar timestamps.
            src_latest_mtime > build_latest_mtime + 10.0
        ):
            current_src_hash = _compute_src_hash(client_dir / "src")
            last_built_src_hash = _read_last_built_src_hash()
            if (
                last_built_src_hash is not None
                and current_src_hash == last_built_src_hash
            ):
                import rich

                rich.print(
                    "[bold](viser)[/bold] Source mtimes changed but content hash matches last build. Skipping rebuild."
                )
                build = False
            else:
                import rich

                rich.print(
                    "[bold](viser)[/bold] Client build looks out of date. Building now..."
                )
                build = True

    # Install nodejs and build if necessary. We assume bash is installed.
    if build:
        _build_viser_client(out_dir=build_dir, cached=False)


def _build_viser_client(out_dir: Path, cached: bool = True) -> None:
    """Create a build of the Viser client.

    Args:
        out_dir: The directory to write the built client to.
        cached: If True, skip the build if the client is already built.
            Instead, we'll simply copy the previous build to the new location.
    """

    if cached and build_dir.exists() and (build_dir / "index.html").exists():
        import rich

        rich.print(
            f"[bold](viser)[/bold] Copying client build from {build_dir} to {out_dir}."
        )
        shutil.copytree(build_dir, out_dir)
        return

    node_bin_dir = _install_sandboxed_node()
    npx_path = node_bin_dir / "npx.cmd" if sys.platform == "win32" else node_bin_dir / "npx"
    print(f"[viser] Using npx at {npx_path}")
    subprocess_env = os.environ.copy()
    subprocess_env["NODE_VIRTUAL_ENV"] = str(node_bin_dir.parent)
    subprocess_env["PATH"] = (
        str(node_bin_dir)
        + (";" if sys.platform == "win32" else ":")
        + subprocess_env["PATH"]
    )
    npm_path = node_bin_dir / "npm.cmd" if sys.platform == "win32" else node_bin_dir / "npm"
    print(f"[viser] Using npm at {npm_path}")
    subprocess.run(
        args=[str(npm_path), "install", "--legacy-peer-deps"],
        env=subprocess_env,
        cwd=client_dir,
        check=False,
    )
    vite_result = subprocess.run(
        args=[
            str(npx_path),
            "vite",
            "build",
            "--base",
            "./",
            "--outDir",
            # Relative path needs to be made absolute, since we change the CWD.
            str(out_dir.absolute()),
        ],
        env=subprocess_env,
        cwd=client_dir,
        check=False,
    )
    if vite_result.returncode == 0:
        _write_last_built_src_hash(_compute_src_hash(client_dir / "src"))


def build_client_entrypoint() -> None:
    """Build the Viser client entrypoint, which is used to launch the viewer."""
    parser = argparse.ArgumentParser(description="Build the Viser client.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--no-cached",
        action="store_false",
        help="If set, skip the build if the client is already built.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else build_dir
    _build_viser_client(out_dir=out_dir, cached=args.no_cached)


def _install_sandboxed_node() -> Path:
    """Install a sandboxed copy of nodejs using nodeenv, and return a path to the
    environment's bin directory (`.nodeenv/bin` or `.nodeenv/Scripts`).

    On Windows, the `.nodeenv/bin` does not exist. Instead, executables are
    installed to `.nodeenv/Scripts`."""

    min_node_version = (20, 19, 0)

    def get_node_bin_dir() -> Path:
        env_dir = client_dir / ".nodeenv"
        node_bin_dir = env_dir / "bin"
        if not node_bin_dir.exists():
            node_bin_dir = env_dir / "Scripts"
        return node_bin_dir

    def parse_node_version(raw: str) -> tuple[int, int, int] | None:
        raw = raw.strip()
        if raw.startswith("v"):
            raw = raw[1:]
        parts = raw.split(".")
        if len(parts) < 3:
            return None
        try:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            return None

    def needs_node_upgrade(node_bin_dir: Path) -> bool:
        node_path = node_bin_dir / ("node.exe" if sys.platform == "win32" else "node")
        if not node_path.exists():
            return True
        try:
            result = subprocess.run(
                [str(node_path), "--version"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return True
        node_version = parse_node_version(result.stdout)
        if node_version is None:
            return True
        return node_version < min_node_version

    node_bin_dir = get_node_bin_dir()
    env_dir = client_dir / ".nodeenv"
    if (node_bin_dir / "npx").exists() and not needs_node_upgrade(node_bin_dir):
        import rich

        rich.print("[bold](viser)[/bold] nodejs is set up!")
        return node_bin_dir
    if env_dir.exists():
        shutil.rmtree(env_dir)

    result = subprocess.run(
        [sys.executable, "-m", "nodeenv", "--node=20.19.0", env_dir], check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install Node.js using nodeenv. "
            "To rebuild the Viser client, install nodeenv with: "
            "pip install 'nodeenv>=1.9.1'"
        )

    node_bin_dir = get_node_bin_dir()
    assert (node_bin_dir / "npx").exists()
    return node_bin_dir


def _modified_time_recursive(dir: Path) -> tuple[float, Path | None]:
    """Recursively get the last modified file timestamp and file path."""
    # Only compare file mtimes. Directory mtimes can change due to transient
    # file-system operations (temp files, editor swaps), which causes
    # intermittent false positives in the "build looks out of date" check.
    latest_mtime = 0.0
    latest_file: Path | None = None
    for f in dir.glob("**/*"):
        if not f.is_file():
            continue
        mtime = f.stat().st_mtime
        if mtime >= latest_mtime:
            latest_mtime = mtime
            latest_file = f
    return latest_mtime, latest_file


def _compute_src_hash(src_dir: Path) -> str:
    hasher = hashlib.sha256()
    for f in sorted(src_dir.glob("**/*")):
        if not f.is_file():
            continue
        rel_path = f.relative_to(src_dir).as_posix()
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(b"\0")
        with f.open("rb") as fp:
            while True:
                chunk = fp.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        hasher.update(b"\0")
    return hasher.hexdigest()


def _read_last_built_src_hash() -> str | None:
    if not _SRC_HASH_MARKER.exists():
        return None
    try:
        return _SRC_HASH_MARKER.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _write_last_built_src_hash(src_hash: str) -> None:
    try:
        _SRC_HASH_MARKER.write_text(src_hash + "\n", encoding="utf-8")
    except OSError:
        pass
