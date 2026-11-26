#!/usr/bin/env python3
"""Build a self-contained, drag-to-Applications macOS Apple Silicon ``.dmg`` for Jasna.

Bundles everything required to run Jasna out of the box on Apple Silicon Macs without
requiring Python, Homebrew, or ``ffmpeg`` to be preinstalled:

- Embedded ARM64 ``Python.framework`` (3.12) + Tcl/Tk 9.1 + ``_tkinter``
- Relocated runtime dylibs inside ``Contents/Frameworks/`` (zero ``/opt/homebrew`` refs)
- Relocated ``ffmpeg`` and ``ffprobe`` inside ``Contents/Resources/tools/``
- Pruned runtime ``site-packages`` (PyTorch MPS, torchvision, PyAV, RF-DETR, ultralytics,
  diffusers, transformers, CustomTkinter, tkinterdnd2 arm64, etc.)
- Clean ``jasna`` source package (explicitly excluding ``jasna/protection``)
- Official open-source weights (BasicVSR++, Lada YOLO v4, Jasna RF-DETR v6)
- Ad-hoc bottom-up ``codesign`` for Apple Silicon execution
- Compressed UDZO disk image ``dist/Jasna-<version>-macos-arm64.dmg`` with an
  ``/Applications`` drop target and matching ``.sha256`` checksum file.

Usage:
    .venv/bin/python scripts/build_macos_dmg.py
    .venv/bin/python scripts/build_macos_dmg.py --skip-dmg
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import os
from pathlib import Path
import plistlib
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile

from jasna import __version__

OPEN_SOURCE_WEIGHTS = (
    "lada_mosaic_restoration_model_generic_v1.2.pth",
    "lada_mosaic_detection_model_v4_fast.pt",
    "rfdetr-v6-large.pt",
)

# Dev/build-only packages in .venv/site-packages that are not required at runtime.
EXCLUDED_SITE_PACKAGE_PREFIXES = (
    "__editable__.jasna",
    "__editable___jasna",
    "pytest",
    "_pytest",
    "pluggy",
    "iniconfig",
    "coverage",
    "nuitka",
    "Nuitka",
    "cmake",
    "ninja",
    "skbuild",
    "scikit_build",
    "pip",
    "wheel",
)

# Large repo-only assets that should not be shipped inside the end-user DMG.
EXCLUDED_ASSET_NAMES = frozenset({
    "test_clip1_1080p.mp4",
    "test_clip1_2160p.mp4",
    "benchmark",
})

# Stdlib directories safe to prune from the embedded Python.framework.
PRUNED_STDLIB_DIRS = frozenset({
    "test",
    "idlelib",
    "turtledemo",
    "ensurepip",
    "venv",
    "lib2to3",
})

STANDALONE_LAUNCHER_C = r'''
#include <Python.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void ensure_dir(const char *path) {
    char tmp[PATH_MAX];
    strncpy(tmp, path, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    for (char *p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static void prepend_path(const char *prefix) {
    const char *old_path = getenv("PATH");
    char new_path[PATH_MAX * 2];
    if (old_path && *old_path) {
        snprintf(new_path, sizeof(new_path), "%s:%s", prefix, old_path);
    } else {
        snprintf(new_path, sizeof(new_path), "%s:/usr/bin:/bin:/usr/sbin:/sbin", prefix);
    }
    setenv("PATH", new_path, 1);
}

static void show_error_dialog(const char *message) {
    if (isatty(STDERR_FILENO) || getenv("JASNA_STANDALONE_SELFTEST") != NULL) {
        fprintf(stderr, "Jasna error: %s\n", message);
        return;
    }
    char cmd[PATH_MAX * 2];
    snprintf(
        cmd,
        sizeof(cmd),
        "/usr/bin/osascript -e 'display dialog \"%s\" with title \"Jasna\" buttons {\"OK\"} default button 1 with icon stop'",
        message
    );
    (void)system(cmd);
}

static int resolve_contents_dir(char *out, size_t out_size) {
    char exe_buf[PATH_MAX];
    uint32_t size = (uint32_t)sizeof(exe_buf);
    if (_NSGetExecutablePath(exe_buf, &size) != 0) {
        return -1;
    }
    char real_exe[PATH_MAX];
    if (realpath(exe_buf, real_exe) == NULL) {
        return -1;
    }
    /* Expected: <App>.app/Contents/MacOS/Jasna -> go up 2 levels to Contents */
    for (int up = 0; up < 2; ++up) {
        char *slash = strrchr(real_exe, '/');
        if (!slash) {
            return -1;
        }
        *slash = '\0';
    }
    strncpy(out, real_exe, out_size - 1);
    out[out_size - 1] = '\0';
    return 0;
}

int main(int argc, char **argv) {
    char contents[PATH_MAX];
    if (resolve_contents_dir(contents, sizeof(contents)) != 0) {
        show_error_dialog("Could not locate Jasna.app bundle contents.");
        return 1;
    }

    const char *home = getenv("HOME");
    if (!home || !*home) {
        home = "/tmp";
    }

    char py_home[PATH_MAX];
    char py_exe[PATH_MAX];
    char app_dir[PATH_MAX];
    char site_dir[PATH_MAX];
    char py_path[PATH_MAX * 2];
    char tcl_root[PATH_MAX];
    char tcl_lib[PATH_MAX];
    char tk_lib[PATH_MAX];
    char tools_dir[PATH_MAX];
    char bundled_weights[PATH_MAX];

    char user_data[PATH_MAX];
    char config_dir[PATH_MAX];
    char weights_dir[PATH_MAX];
    char cache_hf[PATH_MAX];
    char cache_torch[PATH_MAX];
    char cache_mpl[PATH_MAX];
    char log_dir[PATH_MAX];
    char log_path[PATH_MAX];

    snprintf(py_home, sizeof(py_home), "%s/Frameworks/Python.framework/Versions/3.12", contents);
    snprintf(py_exe, sizeof(py_exe), "%s/bin/python3.12", py_home);
    snprintf(app_dir, sizeof(app_dir), "%s/Resources/app", contents);
    snprintf(site_dir, sizeof(site_dir), "%s/Resources/site-packages", contents);
    snprintf(py_path, sizeof(py_path), "%s:%s", app_dir, site_dir);
    snprintf(tcl_root, sizeof(tcl_root), "%s/Resources/tcl", contents);
    snprintf(tcl_lib, sizeof(tcl_lib), "%s/tcl9.1", tcl_root);
    snprintf(tk_lib, sizeof(tk_lib), "%s/tk9.1", tcl_root);
    snprintf(tools_dir, sizeof(tools_dir), "%s/Resources/tools", contents);
    snprintf(bundled_weights, sizeof(bundled_weights), "%s/model_weights", app_dir);

    snprintf(user_data, sizeof(user_data), "%s/Library/Application Support/Jasna", home);
    snprintf(config_dir, sizeof(config_dir), "%s/config", user_data);
    snprintf(weights_dir, sizeof(weights_dir), "%s/model_weights", user_data);
    snprintf(cache_hf, sizeof(cache_hf), "%s/cache/huggingface", user_data);
    snprintf(cache_torch, sizeof(cache_torch), "%s/cache/torch", user_data);
    snprintf(cache_mpl, sizeof(cache_mpl), "%s/cache/matplotlib", user_data);
    snprintf(log_dir, sizeof(log_dir), "%s/logs", user_data);
    snprintf(log_path, sizeof(log_path), "%s/jasna-app.log", log_dir);

    ensure_dir(user_data);
    ensure_dir(config_dir);
    ensure_dir(weights_dir);
    ensure_dir(cache_hf);
    ensure_dir(cache_torch);
    ensure_dir(cache_mpl);
    ensure_dir(log_dir);

    /* Run with a guaranteed writable working directory outside the .app bundle. */
    (void)chdir(user_data);

    prepend_path(tools_dir);
    setenv("PYTHONHOME", py_home, 1);
    setenv("PYTHONPATH", py_path, 1);
    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("PYTHONUTF8", "1", 1);
    setenv("TCL_LIBRARY", tcl_lib, 1);
    setenv("TK_LIBRARY", tk_lib, 1);
    setenv("TCLLIBPATH", tcl_root, 1);
    setenv("JASNA_TOOLS_DIR", tools_dir, 1);
    setenv("JASNA_BUNDLED_WEIGHTS_DIR", bundled_weights, 1);
    setenv("JASNA_MODEL_WEIGHTS_DIR", weights_dir, 0);
    setenv("XDG_CONFIG_HOME", config_dir, 0);
    setenv("HF_HOME", cache_hf, 0);
    setenv("TORCH_HOME", cache_torch, 0);
    setenv("MPLCONFIGDIR", cache_mpl, 0);
    setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1", 0);

    /* Capture Finder/Dock stdout+stderr into Application Support/Jasna/logs/jasna-app.log. */
    if (!isatty(STDERR_FILENO) && getenv("JASNA_NO_LOG_REDIRECT") == NULL && getenv("JASNA_STANDALONE_SELFTEST") == NULL) {
        int log_fd = open(log_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (log_fd >= 0) {
            dup2(log_fd, STDOUT_FILENO);
            dup2(log_fd, STDERR_FILENO);
            close(log_fd);
        }
    }

    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.parse_argv = 0;
    config.user_site_directory = 0;

    status = PyConfig_SetBytesString(&config, &config.home, py_home);
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        show_error_dialog("Failed to set embedded Python home.");
        return 1;
    }

    status = PyConfig_SetBytesString(&config, &config.executable, py_exe);
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        show_error_dialog("Failed to set embedded Python executable.");
        return 1;
    }

    char **filtered_argv = (char **)calloc((size_t)argc + 1, sizeof(char *));
    int filtered_argc = 0;
    filtered_argv[filtered_argc++] = argv[0];
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "-psn_", 5) != 0) {
            filtered_argv[filtered_argc++] = argv[i];
        }
    }
    status = PyConfig_SetBytesArgv(&config, filtered_argc, filtered_argv);
    free(filtered_argv);
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        show_error_dialog("Failed to pass launch arguments to Python.");
        return 1;
    }

    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        show_error_dialog("Failed to initialize embedded Python runtime. Check ~/Library/Application Support/Jasna/logs/jasna-app.log.");
        return 1;
    }

    const char *boot_code =
        "import json, os, sys, traceback\n"
        "from pathlib import Path\n"
        "try:\n"
        "    try:\n"
        "        import certifi\n"
        "        ca = certifi.where()\n"
        "        os.environ.setdefault('SSL_CERT_FILE', ca)\n"
        "        os.environ.setdefault('REQUESTS_CA_BUNDLE', ca)\n"
        "    except Exception:\n"
        "        pass\n"
        "    bundled = Path(os.environ.get('JASNA_BUNDLED_WEIGHTS_DIR', ''))\n"
        "    target = Path(os.environ.get('JASNA_MODEL_WEIGHTS_DIR', ''))\n"
        "    if bundled.is_dir() and str(target):\n"
        "        target.mkdir(parents=True, exist_ok=True)\n"
        "        for src in sorted(bundled.iterdir()):\n"
        "            if not src.is_file() or src.name.startswith('.'):\n"
        "                continue\n"
        "            dst = target / src.name\n"
        "            if dst.is_symlink():\n"
        "                link_target = os.readlink(dst)\n"
        "                if not dst.exists() or '/Contents/Resources/app/model_weights/' in link_target:\n"
        "                    if link_target != str(src):\n"
        "                        dst.unlink()\n"
        "                        dst.symlink_to(src)\n"
        "            elif not dst.exists():\n"
        "                dst.symlink_to(src)\n"
        "    if os.environ.get('JASNA_STANDALONE_SELFTEST') == '1':\n"
        "        import subprocess, tkinter, customtkinter, tkinterdnd2, av, cv2, torch, torchvision, rfdetr, ultralytics\n"
        "        from jasna._frozen import is_frozen\n"
        "        from jasna.engine_paths import model_weights_dir\n"
        "        from jasna.gui.branding import brand_asset_path, LOGO_PNG\n"
        "        from jasna.gui.system_stats import read_gpu_vram\n"
        "        from jasna.license_api import protection_available\n"
        "        from jasna.os_utils import find_executable\n"
        "        ffmpeg = find_executable('ffmpeg')\n"
        "        ffprobe = find_executable('ffprobe')\n"
        "        ff_rc = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True, check=True).returncode\n"
        "        fp_rc = subprocess.run([ffprobe, '-version'], capture_output=True, text=True, check=True).returncode\n"
        "        gpu_pct, vram_pct = read_gpu_vram()\n"
        "        report = {\n"
        "            'executable': sys.executable,\n"
        "            'prefix': sys.prefix,\n"
        "            'is_frozen': is_frozen(),\n"
        "            'protection_available': protection_available(),\n"
        "            'mps_available': bool(torch.backends.mps.is_available()),\n"
        "            'ffmpeg': ffmpeg,\n"
        "            'ffprobe': ffprobe,\n"
        "            'ffmpeg_rc': ff_rc,\n"
        "            'ffprobe_rc': fp_rc,\n"
        "            'weights_dir': str(model_weights_dir()),\n"
        "            'weights': sorted(p.name for p in model_weights_dir().iterdir() if p.exists()),\n"
        "            'logo_exists': brand_asset_path(LOGO_PNG).is_file(),\n"
        "            'gpu_pct': gpu_pct,\n"
        "            'vram_pct': vram_pct,\n"
        "            'tk_version': str(tkinter.TkVersion),\n"
        "        }\n"
        "        print(json.dumps(report, indent=2))\n"
        "    elif len(sys.argv) > 1 and sys.argv[1] in ('--cli', '-h', '--help', '--version'):\n"
        "        if sys.argv[1] == '--cli':\n"
        "            sys.argv.pop(1)\n"
        "        from jasna.main import main\n"
        "        main()\n"
        "    else:\n"
        "        from jasna.gui.app import run_gui\n"
        "        run_gui()\n"
        "except SystemExit:\n"
        "    raise\n"
        "except Exception:\n"
        "    traceback.print_exc()\n"
        "    raise\n";

    int rc = PyRun_SimpleString(boot_code);
    if (Py_FinalizeEx() < 0 && rc == 0) {
        rc = 1;
    }
    if (rc != 0) {
        show_error_dialog("Jasna exited unexpectedly. See ~/Library/Application Support/Jasna/logs/jasna-app.log for details.");
    }
    return rc;
}
'''


def _run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _make_writable(path: Path) -> None:
    if path.is_symlink() or not path.exists():
        return
    mode = path.stat().st_mode
    if not (mode & stat.S_IWUSR):
        path.chmod(mode | stat.S_IWUSR)


def _is_macho(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            magic = f.read(4)
    except OSError:
        return False
    return magic in (
        b"\xcf\xfa\xed\xfe",  # MH_MAGIC_64
        b"\xfe\xed\xfa\xcf",  # MH_CIGAM_64
        b"\xca\xfe\xba\xbe",  # FAT_MAGIC
        b"\xbe\xba\xfe\xca",  # FAT_CIGAM
        b"\xca\xfe\xba\xbf",  # FAT_MAGIC_64
        b"\xbf\xba\xfe\xca",  # FAT_CIGAM_64
    )


def _otool_deps(path: Path) -> list[str]:
    proc = _run(["otool", "-L", str(path)], capture=True)
    deps: list[str] = []
    seen: set[str] = set()
    first_in_section = False
    for line in proc.stdout.splitlines():
        if not line.startswith("\t"):
            first_in_section = True
            continue
        dep = line.strip().split(" (", 1)[0]
        if first_in_section:
            first_in_section = False
            if path.suffix == ".dylib" and os.path.basename(dep) == path.name:
                continue
        if dep not in seen:
            seen.add(dep)
            deps.append(dep)
    return deps


def _otool_rpaths(path: Path) -> list[str]:
    proc = _run(["otool", "-l", str(path)], capture=True)
    rpaths: list[str] = []
    lines = proc.stdout.splitlines()
    for i, line in enumerate(lines):
        if "cmd LC_RPATH" in line:
            for j in range(i + 1, min(i + 6, len(lines))):
                s = lines[j].strip()
                if s.startswith("path "):
                    rpaths.append(s.split(" ", 1)[1].split(" (offset", 1)[0])
                    break
    return rpaths


def _is_system_dep(dep: str) -> bool:
    return dep.startswith("/System/") or dep.startswith("/usr/lib/")


def _resolve_dep_source(dep: str, owner_real: Path) -> Path | None:
    if _is_system_dep(dep):
        return None
    if dep.startswith("/"):
        p = Path(dep)
        return p.resolve() if p.exists() else None
    if dep.startswith("@loader_path/"):
        p = owner_real.parent / dep.removeprefix("@loader_path/")
        return p.resolve() if p.exists() else None
    if dep.startswith("@executable_path/"):
        p = owner_real.parent / dep.removeprefix("@executable_path/")
        return p.resolve() if p.exists() else None
    if dep.startswith("@rpath/"):
        rel = dep.removeprefix("@rpath/")
        candidates: list[Path] = []
        for rp in _otool_rpaths(owner_real):
            if rp.startswith("@loader_path"):
                rp_dir = owner_real.parent / rp.removeprefix("@loader_path").lstrip("/")
            elif rp.startswith("@executable_path"):
                rp_dir = owner_real.parent / rp.removeprefix("@executable_path").lstrip("/")
            else:
                rp_dir = Path(rp)
            candidates.append(rp_dir / rel)
        candidates.append(owner_real.parent / rel)
        candidates.append(Path("/opt/homebrew/lib") / rel)
        for cand in candidates:
            if cand.exists():
                return cand.resolve()
    return None


def _build_icns(logo_png: Path, out_icns: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="jasna_iconset_") as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        sizes = (
            (16, "icon_16x16.png"),
            (32, "icon_16x16@2x.png"),
            (32, "icon_32x32.png"),
            (64, "icon_32x32@2x.png"),
            (128, "icon_128x128.png"),
            (256, "icon_128x128@2x.png"),
            (256, "icon_256x256.png"),
            (512, "icon_256x256@2x.png"),
            (512, "icon_512x512.png"),
            (1024, "icon_512x512@2x.png"),
        )
        for px, filename in sizes:
            subprocess.run(
                ["sips", "-z", str(px), str(px), str(logo_png), "--out", str(iconset / filename)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        _run(["iconutil", "-c", "icns", str(iconset), "-o", str(out_icns)])


def _write_info_plist(plist_path: Path, version: str) -> None:
    info = {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": "Jasna",
        "CFBundleExecutable": "Jasna",
        "CFBundleIconFile": "AppIcon",
        "CFBundleIdentifier": "com.jasna.macos.app",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleName": "Jasna",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSApplicationCategoryType": "public.app-category.video",
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "NSPrincipalClass": "NSApplication",
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Video",
                "CFBundleTypeRole": "Viewer",
                "CFBundleTypeExtensions": ["mp4", "mkv", "mov", "avi", "webm", "m4v", "ts"],
            },
            {
                "CFBundleTypeName": "Image",
                "CFBundleTypeRole": "Viewer",
                "CFBundleTypeExtensions": ["png", "jpg", "jpeg", "webp", "bmp"],
            },
        ],
    }
    with plist_path.open("wb") as f:
        plistlib.dump(info, f)


def _copy_python_framework(dest_frameworks: Path) -> tuple[Path, Path]:
    """Copy Python.framework (3.12) + _tkinter into ``Contents/Frameworks/Python.framework``."""
    framework_prefix = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")
    ldlibrary = sysconfig.get_config_var("LDLIBRARY")
    if not framework_prefix or not ldlibrary:
        raise RuntimeError("Python.framework not found; run with Homebrew framework Python 3.12.")

    src_fw = (Path(framework_prefix) / "Python.framework").resolve()
    src_v312 = (src_fw / "Versions" / "3.12").resolve()
    if not src_v312.is_dir():
        raise FileNotFoundError(f"Expected Python 3.12 framework directory: {src_v312}")

    dest_fw = dest_frameworks / "Python.framework"
    dest_v312 = dest_fw / "Versions" / "3.12"
    dest_v312.mkdir(parents=True)

    # Copy main framework shared library and bin/python3.12.
    py_dylib_dst = dest_v312 / "Python"
    shutil.copy2(src_v312 / "Python", py_dylib_dst)
    _make_writable(py_dylib_dst)

    dest_bin = dest_v312 / "bin"
    dest_bin.mkdir()
    # On macOS framework builds, bin/python3.12 is a posix_spawn trampoline into
    # Resources/Python.app/Contents/MacOS/Python. Copy the real direct interpreter
    # binary so subprocess / multiprocessing invocations via sys.executable work
    # without needing a nested Python.app bundle.
    direct_py = src_v312 / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    py_bin_src = direct_py.resolve() if direct_py.is_file() else (src_v312 / "bin" / "python3.12").resolve()
    py_bin_dst = dest_bin / "python3.12"
    shutil.copy2(py_bin_src, py_bin_dst)
    _make_writable(py_bin_dst)
    py_bin_dst.chmod(0o755)
    (dest_bin / "python3").symlink_to("python3.12")
    (dest_bin / "python").symlink_to("python3.12")

    # Copy Resources/Info.plist (skipping nested Python.app) so codesign treats Python.framework cleanly.
    src_info = src_v312 / "Resources" / "Info.plist"
    if src_info.is_file():
        (dest_v312 / "Resources").mkdir()
        shutil.copy2(src_info, dest_v312 / "Resources" / "Info.plist")

    # Copy stdlib lib/python3.12 (pruning tests, caches, idlelib, site-packages).
    src_stdlib = src_v312 / "lib" / "python3.12"
    dest_stdlib = dest_v312 / "lib" / "python3.12"

    def _ignore_stdlib(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        rel = Path(directory).relative_to(src_stdlib)
        for name in names:
            if name == "__pycache__" or name.endswith((".pyc", ".pyo", ".a")):
                ignored.add(name)
            elif rel == Path(".") and (name in PRUNED_STDLIB_DIRS or name == "site-packages"):
                ignored.add(name)
            elif name == "test" and rel.parts and rel.parts[0] in ("ctypes", "tkinter", "unittest", "sqlite3", "email", "json", "distutils"):
                ignored.add(name)
        return ignored

    shutil.copytree(src_stdlib, dest_stdlib, symlinks=True, ignore=_ignore_stdlib)
    (dest_stdlib / "site-packages").mkdir(exist_ok=True)

    # Copy _tkinter extension from Homebrew python-tk@3.12 into lib-dynload.
    import _tkinter
    src_tkinter = Path(_tkinter.__file__).resolve()
    dst_tkinter = dest_stdlib / "lib-dynload" / src_tkinter.name
    shutil.copy2(src_tkinter, dst_tkinter)
    _make_writable(dst_tkinter)

    # Standard framework symlinks.
    (dest_fw / "Versions" / "Current").symlink_to("3.12")
    (dest_fw / "Python").symlink_to("Versions/Current/Python")
    if (dest_v312 / "Resources").exists():
        (dest_fw / "Resources").symlink_to("Versions/Current/Resources")

    # Fix install_name of Python dylib and python3.12 binary.
    _run(["install_name_tool", "-id", "@rpath/Python.framework/Versions/3.12/Python", str(py_dylib_dst)])
    for dep in _otool_deps(py_bin_dst):
        if dep.endswith("Python.framework/Versions/3.12/Python"):
            _run(["install_name_tool", "-change", dep, "@executable_path/../Python", str(py_bin_dst)])

    return py_dylib_dst, dst_tkinter


def _copy_tcl_tk_scripts(dest_tcl_dir: Path, tkinter_src: Path) -> None:
    """Copy Tcl/Tk 9.1 script libraries next to the app resources."""
    tcl_lib_dir: Path | None = None
    for dep in _otool_deps(tkinter_src):
        if "libtcl" in os.path.basename(dep):
            resolved = Path(dep).resolve()
            tcl_lib_dir = resolved.parent
            break
    if tcl_lib_dir is None:
        tcl_lib_dir = Path("/opt/homebrew/opt/tcl-tk/lib")
    dest_tcl_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("tcl9.1", "tk9.1", "tcl9"):
        src = tcl_lib_dir / sub
        if not src.is_dir():
            raise FileNotFoundError(f"Required Tcl/Tk library directory not found: {src}")
        shutil.copytree(src, dest_tcl_dir / sub, symlinks=True)


def _copy_ffmpeg_tools(dest_tools_dir: Path) -> list[tuple[Path, Path]]:
    """Copy ffmpeg & ffprobe into ``Contents/Resources/tools``; return (dst, original_real_src) pairs."""
    dest_tools_dir.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[Path, Path]] = []
    for name in ("ffmpeg", "ffprobe"):
        found = shutil.which(name) or f"/opt/homebrew/bin/{name}"
        src = Path(found).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Could not find {name} to bundle: {found}")
        dst = dest_tools_dir / name
        shutil.copy2(src, dst)
        _make_writable(dst)
        dst.chmod(0o755)
        copied.append((dst, src))
    return copied


def _relocate_macho_deps(
    seed_targets: list[tuple[Path, Path]],
    frameworks_dir: Path,
) -> int:
    """Recursively bundle all non-system dylibs into ``frameworks_dir`` and rewrite load commands."""
    # Map real source path -> canonical filename stored inside frameworks_dir.
    real_to_name: dict[Path, str] = {}
    # Queue items: (dst_macho_path, original_src_real_path)
    queue: deque[tuple[Path, Path]] = deque(seed_targets)
    processed: set[Path] = set()
    bundled_count = 0

    while queue:
        dst_macho, src_real = queue.popleft()
        if dst_macho in processed:
            continue
        processed.add(dst_macho)
        _make_writable(dst_macho)

        rel_to_fw = os.path.relpath(frameworks_dir, dst_macho.parent)
        deps = _otool_deps(dst_macho)

        for dep in deps:
            if _is_system_dep(dep):
                continue
            if dep.startswith("@rpath/Python.framework/") or dep.startswith("@executable_path/../Python"):
                continue

            resolved = _resolve_dep_source(dep, src_real)
            if resolved is None:
                raise RuntimeError(f"Unable to resolve dylib dependency {dep!r} of {dst_macho} (src={src_real})")

            dep_basename = os.path.basename(dep)
            target_in_fw = frameworks_dir / dep_basename

            if resolved not in real_to_name:
                shutil.copy2(resolved, target_in_fw)
                _make_writable(target_in_fw)
                target_in_fw.chmod(0o755)
                _run(["install_name_tool", "-id", f"@rpath/{dep_basename}", str(target_in_fw)])
                real_to_name[resolved] = dep_basename
                bundled_count += 1
                queue.append((target_in_fw, resolved))
            else:
                canonical = real_to_name[resolved]
                if dep_basename != canonical and not target_in_fw.exists():
                    target_in_fw.symlink_to(canonical)

            if rel_to_fw == ".":
                new_ref = f"@loader_path/{dep_basename}"
            else:
                new_ref = f"@loader_path/{rel_to_fw}/{dep_basename}"

            if dep != new_ref:
                _run(["install_name_tool", "-change", dep, new_ref, str(dst_macho)])

    return bundled_count


def _copy_site_packages(venv_site: Path, dest_site: Path) -> None:
    """Copy runtime site-packages from ``.venv`` while pruning dev tools and non-macOS binaries."""
    dest_site.mkdir(parents=True, exist_ok=True)

    def _is_excluded_top(name: str) -> bool:
        if name == "__pycache__" or name.endswith((".pyc", ".pyo")):
            return True
        for prefix in EXCLUDED_SITE_PACKAGE_PREFIXES:
            if name == prefix or name.startswith(prefix + "-") or name.startswith(prefix + "."):
                return True
        return False

    def _ignore_pkg(pkg_root: Path):
        def _ignore(directory: str, names: list[str]) -> set[str]:
            ignored: set[str] = set()
            rel = Path(directory).relative_to(pkg_root)
            for name in names:
                if name == "__pycache__" or name.endswith((".pyc", ".pyo")):
                    ignored.add(name)
                elif pkg_root.name == "torch" and rel == Path(".") and name in ("include", "test"):
                    ignored.add(name)
                elif pkg_root.name == "tkinterdnd2" and rel == Path("tkdnd") and not name.startswith("osx-arm64"):
                    if (Path(directory) / name).is_dir():
                        ignored.add(name)
                elif name == "direct_url.json" and pkg_root.name.endswith(".dist-info"):
                    ignored.add(name)
            return ignored
        return _ignore

    for entry in sorted(venv_site.iterdir()):
        if _is_excluded_top(entry.name):
            continue
        dst = dest_site / entry.name
        if entry.is_dir() and not entry.is_symlink():
            shutil.copytree(entry, dst, symlinks=True, ignore=_ignore_pkg(entry))
        else:
            shutil.copy2(entry, dst, follow_symlinks=False)


def _copy_app_sources_and_weights(repo_root: Path, dest_app_dir: Path) -> None:
    """Copy clean ``jasna/`` (excluding ``protection/``), ``assets/``, and open-source weights."""
    dest_app_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy jasna/ source tree — EXPLICITLY exclude jasna/protection!
    src_jasna = repo_root / "jasna"

    def _ignore_jasna(directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        rel = Path(directory).relative_to(src_jasna)
        for name in names:
            if name == "__pycache__" or name.endswith((".pyc", ".pyo")):
                ignored.add(name)
            elif rel == Path(".") and name == "protection":
                ignored.add(name)
            elif rel == Path(".") and name == "settings.json":
                ignored.add(name)
        return ignored

    shutil.copytree(src_jasna, dest_app_dir / "jasna", symlinks=False, ignore=_ignore_jasna)

    # Safety assertion: ensure zero protection files made it in.
    if (dest_app_dir / "jasna" / "protection").exists():
        raise RuntimeError("SECURITY CHECK FAILED: jasna/protection must never be bundled.")

    # 2. Copy assets/ (excluding large test videos & benchmarks).
    src_assets = repo_root / "assets"

    def _ignore_assets(directory: str, names: list[str]) -> set[str]:
        if Path(directory) == src_assets:
            return {n for n in names if n in EXCLUDED_ASSET_NAMES}
        return set()

    shutil.copytree(src_assets, dest_app_dir / "assets", symlinks=False, ignore=_ignore_assets)

    # 3. Copy official open-source model weights.
    dest_weights = dest_app_dir / "model_weights"
    dest_weights.mkdir(parents=True, exist_ok=True)
    src_weights = repo_root / "model_weights"
    for filename in OPEN_SOURCE_WEIGHTS:
        src = src_weights / filename
        if not src.is_file():
            raise FileNotFoundError(
                f"Required open-source model weight missing: {src}. "
                "Download official open-source weights before building the standalone DMG."
            )
        shutil.copy2(src, dest_weights / filename)


def _compile_standalone_launcher(out_bin: Path, bundled_py_dylib: Path) -> None:
    """Compile ARM64 Mach-O launcher linked against the bundled Python.framework."""
    include_dir = sysconfig.get_config_var("INCLUDEPY")
    if not include_dir:
        raise RuntimeError("INCLUDEPY not available in sysconfig.")

    with tempfile.TemporaryDirectory(prefix="jasna_standalone_launcher_") as tmp:
        src = Path(tmp) / "launcher.c"
        src.write_text(STANDALONE_LAUNCHER_C, encoding="utf-8")
        cmd = [
            "clang",
            "-arch", "arm64",
            "-O2",
            "-Wall",
            "-Wextra",
            f"-I{include_dir}",
            str(src),
            str(bundled_py_dylib),
            "-Wl,-rpath,@executable_path/../Frameworks",
            "-framework", "CoreFoundation",
            "-o", str(out_bin),
        ]
        _run(cmd)
    out_bin.chmod(0o755)


def _audit_bundle_macho(app_path: Path) -> None:
    """Verify ZERO external Homebrew or user-home paths remain in any Mach-O inside ``app_path``."""
    violations: list[str] = []
    forbidden_prefixes = ("/opt/homebrew", "/usr/local", "/Users/")

    for path in app_path.rglob("*"):
        if not _is_macho(path):
            continue
        for dep in _otool_deps(path):
            if dep.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(app_path)} -> {dep}")

    if violations:
        details = "\n".join(violations[:30])
        raise RuntimeError(
            f"Bundle portability audit FAILED ({len(violations)} external dylib references):\n{details}"
        )


def _codesign_bundle(app_path: Path) -> None:
    """Ad-hoc codesign every Mach-O file bottom-up so Apple Silicon runs the relocated bundle."""
    macho_files: list[Path] = []
    main_exe = app_path / "Contents" / "MacOS" / "Jasna"
    py_fw = app_path / "Contents" / "Frameworks" / "Python.framework"

    for path in app_path.rglob("*"):
        if path == main_exe:
            continue
        if _is_macho(path):
            _make_writable(path)
            macho_files.append(path)

    # Sign deepest binaries first (dylibs/so/tools/python3.12), then Python.framework, then main exe & app.
    macho_files.sort(key=lambda p: len(p.parts), reverse=True)
    for idx, macho in enumerate(macho_files, 1):
        _run(["codesign", "--force", "--sign", "-", str(macho)])
        if idx % 100 == 0 or idx == len(macho_files):
            print(f"  signed {idx}/{len(macho_files)} Mach-O binaries...")

    _run(["codesign", "--force", "--sign", "-", str(py_fw)])
    _run(["codesign", "--force", "--sign", "-", str(main_exe)])
    _run(["codesign", "--force", "--sign", "-", str(app_path)])
    _run(["codesign", "--verify", "--deep", "--strict", str(app_path)])


def _create_dmg(app_path: Path, dmg_path: Path, version: str) -> Path:
    """Package ``Jasna.app`` + ``/Applications`` symlink into a compressed UDZO disk image."""
    dmg_path.parent.mkdir(parents=True, exist_ok=True)
    if dmg_path.exists():
        dmg_path.unlink()

    with tempfile.TemporaryDirectory(prefix="jasna_dmg_stage_") as tmp:
        stage = Path(tmp) / "stage"
        stage.mkdir()

        # Use ditto to preserve symlinks, permissions, and code signatures cleanly.
        _run(["ditto", str(app_path), str(stage / app_path.name)])
        (stage / "Applications").symlink_to("/Applications")

        # Set custom volume icon if SetFile is available (Xcode CLT).
        icns = app_path / "Contents" / "Resources" / "AppIcon.icns"
        setfile = shutil.which("SetFile")
        if icns.is_file() and setfile:
            shutil.copy2(icns, stage / ".VolumeIcon.icns")
            subprocess.run([setfile, "-a", "C", str(stage)], check=False)

        _run([
            "hdiutil", "create",
            "-volname", f"Jasna {version}",
            "-srcfolder", str(stage),
            "-ov",
            "-format", "UDZO",
            "-imagekey", "zlib-level=6",
            str(dmg_path),
        ])

    sha256 = hashlib.sha256()
    with dmg_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    checksum_path = dmg_path.with_name(dmg_path.name + ".sha256")
    checksum_path.write_text(f"{sha256.hexdigest()}  {dmg_path.name}\n", encoding="utf-8")
    return checksum_path


def build_standalone_app(repo_root: Path, output_app: Path) -> Path:
    logo_png = repo_root / "assets" / "jasna-logo.png"
    if not logo_png.is_file():
        raise FileNotFoundError(f"Logo asset not found: {logo_png}")

    venv_site = repo_root / ".venv" / "lib" / "python3.12" / "site-packages"
    if not venv_site.is_dir():
        raise FileNotFoundError(f"Expected .venv site-packages at {venv_site}")

    if output_app.exists():
        shutil.rmtree(output_app)

    contents = output_app / "Contents"
    macos_dir = contents / "MacOS"
    frameworks_dir = contents / "Frameworks"
    resources_dir = contents / "Resources"
    macos_dir.mkdir(parents=True)
    frameworks_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)

    print("[1/7] Writing bundle metadata and app icon...")
    _write_info_plist(contents / "Info.plist", __version__)
    (contents / "PkgInfo").write_text("APPL????", encoding="ascii")
    _build_icns(logo_png, resources_dir / "AppIcon.icns")

    print("[2/7] Embedding Python.framework 3.12 and _tkinter...")
    py_dylib_dst, dst_tkinter = _copy_python_framework(frameworks_dir)

    print("[3/7] Copying Tcl/Tk 9.1 scripts and ffmpeg/ffprobe...")
    import _tkinter
    src_tkinter = Path(_tkinter.__file__).resolve()
    _copy_tcl_tk_scripts(resources_dir / "tcl", src_tkinter)
    copied_tools = _copy_ffmpeg_tools(resources_dir / "tools")

    print("[4/7] Relocating Homebrew dylibs into Contents/Frameworks/...")
    seed_targets: list[tuple[Path, Path]] = []
    # Seed all lib-dynload/*.so (matching original Homebrew paths for relative lookup).
    src_dynload = Path(sysconfig.get_config_var("DESTSHARED")).resolve()
    dst_dynload = frameworks_dir / "Python.framework" / "Versions" / "3.12" / "lib" / "python3.12" / "lib-dynload"
    for so_file in sorted(dst_dynload.glob("*.so")):
        if so_file == dst_tkinter:
            seed_targets.append((so_file, src_tkinter))
        else:
            orig = (src_dynload / so_file.name).resolve()
            seed_targets.append((so_file, orig if orig.exists() else so_file))
    seed_targets.extend(copied_tools)

    bundled_dylibs = _relocate_macho_deps(seed_targets, frameworks_dir)
    print(f"  bundled and relocated {bundled_dylibs} dylibs.")

    print("[5/7] Copying pruned runtime site-packages, jasna source, assets, and open-source weights...")
    _copy_site_packages(venv_site, resources_dir / "site-packages")
    _copy_app_sources_and_weights(repo_root, resources_dir / "app")

    print("[6/7] Compiling standalone ARM64 launcher...")
    _compile_standalone_launcher(macos_dir / "Jasna", py_dylib_dst)

    print("[7/7] Auditing bundle portability and ad-hoc signing...")
    _audit_bundle_macho(output_app)
    _codesign_bundle(output_app)
    return output_app


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("scripts/build_macos_dmg.py can only run on macOS.")
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build standalone macOS ARM64 Jasna.app and .dmg installer.")
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=repo_root / "dist",
        help="Output directory for standalone Jasna.app and .dmg (default: <repo>/dist)",
    )
    parser.add_argument(
        "--skip-dmg",
        action="store_true",
        help="Only build standalone dist/Jasna.app without creating the .dmg.",
    )
    args = parser.parse_args()

    dist_dir = args.dist_dir.resolve()
    dist_dir.mkdir(parents=True, exist_ok=True)
    app_path = dist_dir / "Jasna.app"
    build_standalone_app(repo_root, app_path)
    print(f"Built standalone app: {app_path}")

    if not args.skip_dmg:
        dmg_name = f"Jasna-{__version__}-macos-arm64.dmg"
        dmg_path = dist_dir / dmg_name
        print(f"Creating compressed DMG: {dmg_path} ...")
        checksum_path = _create_dmg(app_path, dmg_path, __version__)
        size_mb = dmg_path.stat().st_size / (1024 * 1024)
        print(f"Built DMG: {dmg_path} ({size_mb:.1f} MB)")
        print(f"SHA-256:   {checksum_path.read_text(encoding='utf-8').strip()}")


if __name__ == "__main__":
    main()
