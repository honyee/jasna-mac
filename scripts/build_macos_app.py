#!/usr/bin/env python3
"""Build a native, clickable macOS ``Jasna.app`` bundle for Apple Silicon.

Compiles a lightweight ARM64 Mach-O launcher linked directly against the active
venv's Python.framework so macOS LaunchServices, Dock, menu bar, Dark Mode, and
TCC permissions all identify the running process as ``Jasna.app`` (rather than
a generic ``Python`` binary), while running the editable ``jasna`` codebase and
Metal/MPS PyTorch environment inside ``.venv`` with zero gigabyte duplication.

Usage:
    .venv/bin/python scripts/build_macos_app.py
"""
from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from jasna import __version__

LAUNCHER_C = r'''
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

#ifndef FALLBACK_PROJECT_ROOT
#error "FALLBACK_PROJECT_ROOT must be defined at compile time"
#endif

static int dir_has_venv(const char *dir) {
    char probe[PATH_MAX];
    int n = snprintf(probe, sizeof(probe), "%s/.venv/bin/python", dir);
    return n > 0 && n < (int)sizeof(probe) && access(probe, X_OK) == 0;
}

static void resolve_project_root(char *out, size_t out_size) {
    char exe_buf[PATH_MAX];
    uint32_t size = (uint32_t)sizeof(exe_buf);
    if (_NSGetExecutablePath(exe_buf, &size) == 0) {
        char real_exe[PATH_MAX];
        if (realpath(exe_buf, real_exe) != NULL) {
            /* Expected layout: <project_root>/Jasna.app/Contents/MacOS/Jasna */
            char candidate[PATH_MAX];
            strncpy(candidate, real_exe, sizeof(candidate) - 1);
            candidate[sizeof(candidate) - 1] = '\0';
            for (int up = 0; up < 4; ++up) {
                char *slash = strrchr(candidate, '/');
                if (!slash) {
                    break;
                }
                *slash = '\0';
            }
            if (dir_has_venv(candidate)) {
                strncpy(out, candidate, out_size - 1);
                out[out_size - 1] = '\0';
                return;
            }
        }
    }
    strncpy(out, FALLBACK_PROJECT_ROOT, out_size - 1);
    out[out_size - 1] = '\0';
}

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
        snprintf(new_path, sizeof(new_path), "%s", prefix);
    }
    setenv("PATH", new_path, 1);
}

static void show_error_dialog(const char *message) {
    char cmd[PATH_MAX * 2];
    snprintf(
        cmd,
        sizeof(cmd),
        "/usr/bin/osascript -e 'display dialog \"%s\" with title \"Jasna\" buttons {\"OK\"} default button 1 with icon stop'",
        message
    );
    (void)system(cmd);
}

int main(int argc, char **argv) {
    char root[PATH_MAX];
    resolve_project_root(root, sizeof(root));

    if (!dir_has_venv(root)) {
        show_error_dialog("Could not find Jasna .venv environment. Please rebuild Jasna.app from the project folder.");
        return 1;
    }

    if (chdir(root) != 0) {
        show_error_dialog("Failed to switch to Jasna project directory.");
        return 1;
    }

    char venv_dir[PATH_MAX];
    char venv_bin[PATH_MAX];
    char python_exe[PATH_MAX];
    char config_dir[PATH_MAX];
    char cache_hf[PATH_MAX];
    char cache_torch[PATH_MAX];
    char path_prefix[PATH_MAX + 128];
    char log_path[PATH_MAX];

    snprintf(venv_dir, sizeof(venv_dir), "%s/.venv", root);
    snprintf(venv_bin, sizeof(venv_bin), "%s/.venv/bin", root);
    snprintf(python_exe, sizeof(python_exe), "%s/.venv/bin/python", root);
    snprintf(config_dir, sizeof(config_dir), "%s/.jasna_config", root);
    snprintf(cache_hf, sizeof(cache_hf), "%s/.cache/huggingface", root);
    snprintf(cache_torch, sizeof(cache_torch), "%s/.cache/torch", root);
    snprintf(path_prefix, sizeof(path_prefix), "%s:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin", venv_bin);
    snprintf(log_path, sizeof(log_path), "%s/jasna-app.log", config_dir);

    ensure_dir(config_dir);
    ensure_dir(cache_hf);
    ensure_dir(cache_torch);

    prepend_path(path_prefix);
    setenv("VIRTUAL_ENV", venv_dir, 0);
    setenv("XDG_CONFIG_HOME", config_dir, 0);
    setenv("HF_HOME", cache_hf, 0);
    setenv("TORCH_HOME", cache_torch, 0);
    setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1", 0);
    setenv("PYTHONUTF8", "1", 1);

    /* When launched from Finder/Dock (no TTY), capture stdout/stderr to a log file. */
    if (!isatty(STDERR_FILENO)) {
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

    status = PyConfig_SetBytesString(&config, &config.executable, python_exe);
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        show_error_dialog("Failed to configure embedded Python executable path.");
        return 1;
    }

    /* Filter out macOS Finder -psn_* launch argument before passing argv to Python. */
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
        show_error_dialog("Failed to initialize Python runtime. Check .jasna_config/jasna-app.log.");
        return 1;
    }

    const char *boot_code =
        "import os, sys, traceback\n"
        "try:\n"
        "    from jasna.gui.app import run_gui\n"
        "    run_gui()\n"
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
        show_error_dialog("Jasna exited unexpectedly. See .jasna_config/jasna-app.log for details.");
    }
    return rc;
}
'''


def _build_icns(logo_png: Path, out_icns: Path) -> None:
    """Convert a high-res PNG logo into a multi-resolution Apple ``.icns`` icon."""
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
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(out_icns)],
            check=True,
        )


def _compile_launcher(project_root: Path, out_bin: Path) -> None:
    """Compile the native ARM64 Mach-O launcher linked against Python.framework."""
    include_dir = sysconfig.get_config_var("INCLUDEPY")
    framework_prefix = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX")
    ldlibrary = sysconfig.get_config_var("LDLIBRARY")
    if not include_dir or not framework_prefix or not ldlibrary:
        raise RuntimeError("Python.framework paths not available; use Homebrew/framework Python on macOS.")
    python_dylib = Path(framework_prefix) / ldlibrary
    if not python_dylib.is_file():
        raise FileNotFoundError(f"Python framework binary not found: {python_dylib}")

    with tempfile.TemporaryDirectory(prefix="jasna_launcher_") as tmp:
        src = Path(tmp) / "launcher.c"
        src.write_text(LAUNCHER_C, encoding="utf-8")
        cmd = [
            "clang",
            "-arch", "arm64",
            "-O2",
            "-Wall",
            "-Wextra",
            f"-DFALLBACK_PROJECT_ROOT=\"{project_root}\"",
            f"-I{include_dir}",
            str(src),
            str(python_dylib),
            "-Wl,-rpath", str(python_dylib.parent),
            "-framework", "CoreFoundation",
            "-o", str(out_bin),
        ]
        subprocess.run(cmd, check=True)
    out_bin.chmod(0o755)


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


def build_app(project_root: Path, output_app: Path) -> Path:
    logo_png = project_root / "assets" / "jasna-logo.png"
    if not logo_png.is_file():
        raise FileNotFoundError(f"Logo asset not found: {logo_png}")

    if output_app.exists():
        shutil.rmtree(output_app)

    macos_dir = output_app / "Contents" / "MacOS"
    resources_dir = output_app / "Contents" / "Resources"
    macos_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)

    _write_info_plist(output_app / "Contents" / "Info.plist", __version__)
    (output_app / "Contents" / "PkgInfo").write_text("APPL????", encoding="ascii")
    _build_icns(logo_png, resources_dir / "AppIcon.icns")
    _compile_launcher(project_root, macos_dir / "Jasna")

    # Ad-hoc local signature so macOS Gatekeeper & LaunchServices run it cleanly.
    subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(output_app)],
        check=True,
    )
    return output_app


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("scripts/build_macos_app.py can only run on macOS.")
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build clickable Jasna.app bundle for macOS Apple Silicon.")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "Jasna.app",
        help="Output .app bundle path (default: <repo>/Jasna.app)",
    )
    args = parser.parse_args()
    app_path = build_app(repo_root, args.output.resolve())
    print(f"Built macOS application bundle: {app_path}")


if __name__ == "__main__":
    main()
