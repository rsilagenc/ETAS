import argparse
import ast
import csv
import math
import re
from pathlib import Path


ITERATION_RE = re.compile(r"^iteration\s+(\d+)$")
PARAMETER_RE = re.compile(r"""^['"]([^'"]+)['"]:\s*(.*?),?$""")
NUMPY_SCALAR_RE = re.compile(r"np\.[A-Za-z_]\w*\((.*)\)")
FREE_PRODUCTIVITY_RE = re.compile(
    r"free_productivity=(True|False), n_hat=([^,]+), i_hat=(.+)"
)
LOG10_C_DIAGNOSTICS_RE = re.compile(
    r"log10_c initial=([^,]+), final=([^,]+), "
    r"bounds=\[([^,]+),\s*([^\]]+)\], "
    r"distance from lower bound=([^,]+)"
)


def _message_from_line(line):
    if line[:2].isdigit() and " - " in line:
        return line.split(" - ", 1)[1].strip()
    return line.strip()


def _parse_value(value, line_number):
    value = value.removesuffix(",").strip()
    if value == "None":
        return None

    match = NUMPY_SCALAR_RE.fullmatch(value)
    if match:
        value = match.group(1).strip()

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported parameter value on log line {line_number}: {value}"
        ) from exc


def _parse_number(value, line_number):
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported numeric value on log line {line_number}: {value}"
        ) from exc


def _parse_diagnostics(message, line_number):
    if message.startswith("free_productivity="):
        match = FREE_PRODUCTIVITY_RE.fullmatch(message)
        if not match:
            raise ValueError(
                f"Could not parse iteration statistics on log line {line_number}"
            )
        free_productivity, n_hat, i_hat = match.groups()
        return {
            "free_productivity": free_productivity == "True",
            "n_hat": _parse_number(n_hat, line_number),
            "i_hat": (
                None if i_hat.strip() == "None"
                else _parse_number(i_hat, line_number)
            ),
        }

    if message.startswith("Initial objective value: "):
        return {
            "initial_objective": _parse_number(
                message.partition(":")[2].strip(), line_number
            )
        }

    for label, key in (
        ("res.success: ", "res.success"),
        ("res.message: ", "res.message"),
        ("res.fun: ", "res.fun"),
    ):
        if message.startswith(label):
            value = message[len(label):]
            if key == "res.success":
                if value not in ("True", "False"):
                    raise ValueError(
                        f"Invalid res.success on log line {line_number}: "
                        f"{value}"
                    )
                return {key: value == "True"}
            if key == "res.fun":
                return {key: _parse_number(value, line_number)}
            return {key: value}

    if message.startswith("Optimization bounds: "):
        try:
            bounds = ast.literal_eval(message.partition(":")[2].strip())
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Could not parse optimization bounds on log line {line_number}"
            ) from exc
        if not isinstance(bounds, dict):
            raise ValueError(
                f"Optimization bounds are not a dictionary on log line "
                f"{line_number}"
            )

        values = {}
        for name, pair in bounds.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(
                    f"Invalid bounds for {name} on log line {line_number}"
                )
            values[f"bounds.{name}.lower"] = pair[0]
            values[f"bounds.{name}.upper"] = pair[1]
        return values

    match = LOG10_C_DIAGNOSTICS_RE.fullmatch(message)
    if match:
        initial, final, lower, upper, distance = (
            _parse_number(value, line_number) for value in match.groups()
        )
        return {
            "log10_c_initial": initial,
            "log10_c_final": final,
            "log10_c_lower_bound": lower,
            "log10_c_upper_bound": upper,
            "log10_c_distance_from_lower_bound": distance,
            "log10_c_at_lower_bound": math.isclose(
                final, lower, rel_tol=0.0, abs_tol=1e-8
            ),
        }

    return {}


def parse_iteration_data(log_path):
    iterations = {}
    current_iteration = None
    current_parameters = None

    with log_path.open(encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            message = _message_from_line(line)
            stripped = message.strip()
            iteration_match = ITERATION_RE.fullmatch(stripped)
            if iteration_match:
                current_iteration = int(iteration_match.group(1))
                if current_iteration in iterations:
                    raise ValueError(
                        f"Duplicate iteration {current_iteration} in {log_path}"
                    )
                iterations[current_iteration] = {
                    "parameters": {},
                    "diagnostics": {},
                }
                continue

            if "new parameters:" in stripped:
                if current_iteration is None:
                    raise ValueError(
                        f"Found parameters without an iteration on line "
                        f"{line_number}"
                    )
                current_parameters = {}
                continue

            if current_parameters is None:
                if current_iteration is not None:
                    iterations[current_iteration]["diagnostics"].update(
                        _parse_diagnostics(message, line_number)
                    )
                continue

            if stripped.startswith("{"):
                stripped = stripped[1:].strip()
            block_finished = "}" in stripped
            if block_finished:
                stripped = stripped.split("}", 1)[0].strip()

            if stripped:
                parameter_match = PARAMETER_RE.fullmatch(stripped)
                if not parameter_match:
                    raise ValueError(
                        f"Could not parse parameter on log line {line_number}: "
                        f"{stripped}"
                    )
                name, value = parameter_match.groups()
                current_parameters[name] = _parse_value(value, line_number)

            if block_finished:
                if not current_parameters:
                    raise ValueError(
                        f"Empty parameter block ending on line {line_number}"
                    )
                iterations[current_iteration]["parameters"] = current_parameters
                current_parameters = None

    if current_parameters is not None:
        raise ValueError("The log ends before the current parameter block closes")
    if not iterations or any(
        not data["parameters"] for data in iterations.values()
    ):
        raise ValueError(f"No iteration parameter blocks found in {log_path}")

    return iterations


def write_csv(iterations, csv_path):
    ordered_iterations = sorted(iterations)
    parameter_names = list(
        dict.fromkeys(
            name
            for iteration in ordered_iterations
            for name in iterations[iteration]["parameters"]
        )
    )
    diagnostic_names = list(
        dict.fromkeys(
            name
            for iteration in ordered_iterations
            for name in iterations[iteration]["diagnostics"]
        )
    )

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["parameter"]
            + [
                f"iteration{index}"
                for index in range(1, len(ordered_iterations) + 1)
            ]
        )
        for section, names in (
            ("parameters", parameter_names),
            ("diagnostics", diagnostic_names),
        ):
            for name in names:
                writer.writerow(
                    [name]
                    + [
                        iterations[iteration][section].get(name)
                        for iteration in ordered_iterations
                    ]
                )


def main():
    default_log = Path(__file__).with_name("invert_etas.log")
    parser = argparse.ArgumentParser(
        description="Extract iteration parameter values from an ETAS log."
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        type=Path,
        default=default_log,
        help=f"log file to parse (default: {default_log})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="CSV output path (default: <log-name>_iterations.csv)",
    )
    args = parser.parse_args()

    output_path = args.output or args.log_file.with_name(
        f"{args.log_file.stem}_iterations.csv"
    )
    iterations = parse_iteration_data(args.log_file)
    write_csv(iterations, output_path)
    print(f"Wrote {len(iterations)} iterations to {output_path}")


if __name__ == "__main__":
    main()
