"""Render DOCX with the canonical renderer, using an isolated Word COM export.

Usage: bundled-python render_with_word.py INPUT.docx --output_dir QA_DIR --emit_pdf
The managed render_docx.py is imported without changes. Only its DOCX-to-PDF
conversion function is replaced; its DPI calculation and PNG rendering remain.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


RUNTIME = Path(r"C:\Users\agent\.cache\codex-runtimes\codex-primary-runtime\dependencies")
CANONICAL_RENDERER = Path(
    r"C:\Users\agent\.codex\plugins\cache\openai-primary-runtime\documents\26.1011.51657\skills\documents\render_docx.py"
)
POWERSHELL = RUNTIME / "native" / "powershell" / "pwsh.exe"
POPPLER_BIN = RUNTIME / "native" / "poppler" / "Library" / "bin"


WORD_EXPORT_SCRIPT = r'''
param(
    [Parameter(Mandatory=$true)][string]$SourcePath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$wordRenderer = $null
$wordDocument = $null
try {
    $SourcePath = [IO.Path]::GetFullPath($SourcePath)
    $OutputPath = [IO.Path]::GetFullPath($OutputPath)
    # New-Object creates our own automation instance; never attach to an open app.
    $wordRenderer = New-Object -ComObject Word.Application
    $wordRenderer.Visible = $false
    $wordRenderer.DisplayAlerts = 0
    $wordRenderer.AutomationSecurity = 3
    # ConfirmConversions=false, ReadOnly=true, AddToRecentFiles=false.
    $wordDocument = $wordRenderer.Documents.Open($SourcePath, $false, $true, $false)
    $wordDocument.ExportAsFixedFormat($OutputPath, 17, $false)
    if (!(Test-Path -LiteralPath $OutputPath)) {
        throw 'Word did not create the requested PDF.'
    }
    $pdfInfo = Get-Item -LiteralPath $OutputPath
    if ($pdfInfo.Length -le 0) {
        throw 'Word created an empty PDF.'
    }
    Write-Output ('Word {0}: exported read-only input to {1} ({2} bytes)' -f $wordRenderer.Version, $OutputPath, $pdfInfo.Length)
}
finally {
    if ($null -ne $wordDocument) {
        try { $wordDocument.Close([ref]0) } finally {
            [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordDocument)
        }
    }
    if ($null -ne $wordRenderer) {
        try { $wordRenderer.Quit([ref]0) } finally {
            [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordRenderer)
        }
    }
}
'''


def convert_using_word(
    doc_path: str,
    user_profile: str,
    convert_tmp_dir: str,
    stem: str,
    verbose: bool,
) -> tuple[str, str]:
    del user_profile  # A separate Word automation instance owns its document.
    pdf_path = Path(convert_tmp_dir) / f"{stem}.pdf"
    export_script = Path(convert_tmp_dir) / "word_export_pdf.ps1"
    export_script.write_text(WORD_EXPORT_SCRIPT, encoding="utf-8-sig")
    command = [
        str(POWERSHELL), "-NoLogo", "-NoProfile", "-NonInteractive", "-STA",
        "-File", str(export_script), "-SourcePath", str(Path(doc_path).resolve()),
        "-OutputPath", str(pdf_path),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Word PDF export exceeded 180 seconds.") from error
    debug = "\n".join([
        "--- Word COM read-only DOCX->PDF ---",
        "Source: " + str(Path(doc_path).resolve()),
        "Exit: " + str(result.returncode),
        result.stdout.strip(),
        result.stderr.strip(),
    ])
    if verbose:
        print(debug)
    if result.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return str(pdf_path), debug
    return "", debug


def main() -> None:
    if sys.platform != "win32":
        raise RuntimeError("This task-local renderer requires Windows Word COM.")
    for dependency in (CANONICAL_RENDERER, POWERSHELL, POPPLER_BIN):
        if not dependency.exists():
            raise FileNotFoundError(dependency)
    # All PATH changes affect only this helper and its child processes.
    os.environ["PATH"] = str(POPPLER_BIN) + os.pathsep + os.environ.get("PATH", "")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("task_canonical_render_docx", CANONICAL_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the canonical document renderer.")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.convert_to_pdf = convert_using_word
    renderer.main()


if __name__ == "__main__":
    main()
