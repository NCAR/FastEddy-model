#!/usr/bin/env python3

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert gcov JSON output into an LCOV tracefile."
    )
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--object-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gcov", required=True)
    parser.add_argument("--source", action="append", default=[])
    return parser.parse_args()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def normalize_source_path(raw_path: str, source_root: Path) -> Path:
    source_path = Path(raw_path)
    if not source_path.is_absolute():
        source_path = source_root / source_path
    return source_path.resolve()


def run_gcov(gcov_exe: str, temp_dir: Path, object_file: Path, source_file: Path) -> None:
    result = subprocess.run(
        [gcov_exe, "-j", "-b", "-c", "-o", str(object_file), str(source_file)],
        check=False,
        cwd=temp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gcov failed for {source_file} with object {object_file}:\n{result.stdout}"
        )


def collect_trace_data(
    temp_dir: Path, source_root: Path, build_root: Path
) -> dict[Path, dict[str, dict]]:
    trace_data: dict[Path, dict[str, dict]] = {}
    for json_file in temp_dir.glob("*.gcov.json.gz"):
        with gzip.open(json_file, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        for file_data in payload.get("files", []):
            source_path = normalize_source_path(file_data["file"], source_root)
            if not is_relative_to(source_path, source_root):
                continue
            if is_relative_to(source_path, build_root):
                continue

            entry = trace_data.setdefault(
                source_path,
                {"functions": {}, "lines": {}},
            )

            for function in file_data.get("functions", []):
                func_key = (
                    int(function["start_line"]),
                    function.get("demangled_name") or function["name"],
                )
                previous_count = entry["functions"].get(func_key, 0)
                entry["functions"][func_key] = previous_count + int(
                    function.get("execution_count", 0)
                )

            for line in file_data.get("lines", []):
                line_number = int(line["line_number"])
                previous_count = entry["lines"].get(line_number, 0)
                entry["lines"][line_number] = previous_count + int(line.get("count", 0))

    return trace_data


def write_lcov(output_file: Path, trace_data: dict[Path, dict[str, dict]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for source_path in sorted(trace_data):
            entry = trace_data[source_path]
            functions = entry["functions"]
            lines = entry["lines"]

            handle.write("TN:FastEddy\n")
            handle.write(f"SF:{source_path}\n")

            for (start_line, function_name), _ in sorted(functions.items()):
                handle.write(f"FN:{start_line},{function_name}\n")
            for (_, function_name), count in sorted(functions.items()):
                handle.write(f"FNDA:{count},{function_name}\n")

            for line_number, count in sorted(lines.items()):
                handle.write(f"DA:{line_number},{count}\n")

            handle.write(f"FNF:{len(functions)}\n")
            handle.write(f"FNH:{sum(1 for count in functions.values() if count > 0)}\n")
            handle.write(f"LF:{len(lines)}\n")
            handle.write(f"LH:{sum(1 for count in lines.values() if count > 0)}\n")
            handle.write("end_of_record\n")


def main() -> int:
    args = parse_args()
    build_root = Path(args.build_dir).resolve()
    source_root = Path(args.source_dir).resolve()
    object_root = Path(args.object_dir).resolve()
    output_file = Path(args.output).resolve()

    gcov_exe = shutil.which(args.gcov) or args.gcov
    if not shutil.which(gcov_exe) and not Path(gcov_exe).exists():
        raise FileNotFoundError(f"gcov executable not found: {args.gcov}")

    with tempfile.TemporaryDirectory(prefix="fasteddy-gcov-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for source in args.source:
            source_file = Path(source).resolve()
            relative_parent = source_file.relative_to(source_root).parent
            object_dir = object_root / relative_parent
            object_file = object_dir / f"{source_file.name}.o"
            gcno_file = object_dir / f"{source_file.name}.gcno"
            if not gcno_file.exists():
                continue
            run_gcov(gcov_exe, temp_dir, object_file, source_file)

        trace_data = collect_trace_data(temp_dir, source_root, build_root)

    if not trace_data:
        print(
            "No coverage JSON files were generated. "
            "Reconfigure with ENABLE_COVERAGE=ON and run the tests first.",
            file=sys.stderr,
        )
        return 1

    write_lcov(output_file, trace_data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
