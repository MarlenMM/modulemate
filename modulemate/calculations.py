"""Bounded arithmetic and descriptive statistics; no eval, code execution or LLM math."""

from __future__ import annotations

import ast
import math
import operator
import re
import statistics

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
FUNCTIONS = {"sqrt": math.sqrt, "abs": abs, "log": math.log, "log10": math.log10, "round": round}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
# A final 'and' belongs to the same list, not a silently omitted observation.
NUMBER_LIST = rf"{NUMBER}(?:(?:\s*,\s*(?:and\s+)?|\s+and\s+){NUMBER})+"


def arithmetic(expression):
    """A tiny numeric grammar with explicit complexity/magnitude limits."""
    if not isinstance(expression, str) or not 1 <= len(expression) <= 300:
        raise ValueError("Expression limit")
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise ValueError("Expression limit")

    def visit(node, depth=0):
        if depth > 16:
            raise ValueError("Expression depth")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Exponent limit")
            value = OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
            and len(node.args) == 1
            and not node.keywords
        ):
            value = FUNCTIONS[node.func.id](visit(node.args[0], depth + 1))
        else:
            raise ValueError("Unsupported expression")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Numeric limit")
        return value

    return visit(tree.body)


def describe(values):
    if not 1 <= len(values) <= 1000 or any(
        type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e12 for x in values
    ):
        raise ValueError("Data limit")
    ordered = sorted(values)
    n = len(ordered)
    result = {
        "values": values,
        "sorted": ordered,
        "count": n,
        "sum": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "middle_values": ordered[(n - 1) // 2 : n // 2 + 1],
        "modes": statistics.multimode(values),
        "range": max(values) - min(values),
        "population_variance": statistics.pvariance(values),
        "population_standard_deviation": statistics.pstdev(values),
    }
    if n > 1:
        result["sample_variance"] = statistics.variance(values)
        result["sample_standard_deviation"] = statistics.stdev(values)
    return result


def verified_calculations(question):
    number = NUMBER
    results = []
    if re.search(
        r"\b(?:medians?|means?|modes?|variances?|standard deviations?|range|data set|dataset|observations)\b",
        question,
        re.I,
    ):
        if re.search(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?!\d)", question):
            return [
                {
                    "origin": "learner's data formatting",
                    "error": "The commas may be thousands separators or separate observations. Ask for an unambiguous list, such as [1000, 2000], before computing.",
                }
            ]
        matches = list(re.finditer(rf"(?<![\w./])({NUMBER_LIST})(?![\w./])", question))
        if not matches:
            singleton = re.search(
                rf"\b(?:median|mean|modes?|variance|standard deviation|range)\s+(?:of|for)\s*[\[({{]?\s*({number})\s*[\])}}]?[.?! ]*$",
                question,
                re.I,
            )
            matches = [singleton] if singleton else []
        if matches:
            for match in matches:
                values = [float(s) for s in re.findall(number, match[1])]
                try:
                    stats = describe(values)
                except ValueError:
                    results.append(
                        {
                            "origin": "numbers explicitly supplied by the learner",
                            "error": "Numeric tool limit: use 1-1000 finite observations, each with magnitude at most 1e12. No result was computed.",
                        }
                    )
                    continue
                results.append(
                    {
                        "origin": "numbers explicitly supplied by the learner",
                        "input": match[1],
                        "statistics": stats,
                    }
                )
    percent = re.search(rf"({number})\s*(?:%|percent)\s+of\s+({number})", question, re.I)
    if percent:
        expression = f"{percent[1]} / 100 * {percent[2]}"
        results.append(
            {
                "origin": "learner's percentage question",
                "expression": expression,
                "result": arithmetic(expression),
            }
        )
    # Only an explicitly requested arithmetic expression, never source text or
    # natural-language policy numbers, is interpreted as a calculation.
    expr = re.search(
        r"(?:calculate|compute|what is|evaluate)\s+([\d\s()+*/.\-^%]+)(?:[?!.]|$)", question, re.I
    )
    if expr and re.search(r"[+*/^%]|\d\s*-\s*\d", expr[1]):
        try:
            results.append(
                {
                    "origin": "learner's arithmetic expression",
                    "expression": expr[1].strip(),
                    "result": arithmetic(expr[1].strip()),
                }
            )
        except (ValueError, SyntaxError, ArithmeticError, OverflowError):
            pass
    # Unrelated statistics used to distract the critic into alleging claims
    # that were never in the answer. Supply only the requested quantities.
    for result in results:
        if "statistics" not in result:
            continue
        keys = {"values", "sorted", "count"}
        for term, fields in (
            ("median", {"median", "middle_values"}),
            ("mean", {"mean", "sum"}),
            ("mode", {"modes"}),
            ("range", {"range"}),
            ("variance", {"population_variance", "sample_variance"}),
            ("standard deviation", {"population_standard_deviation", "sample_standard_deviation"}),
        ):
            if re.search(r"\b" + term + r"s?\b", question, re.I):
                keys.update(fields)
        result["statistics"] = {k: v for k, v in result["statistics"].items() if k in keys}
        sample = bool(re.search(r"\bsample\b", question, re.I))
        population = bool(re.search(r"\bpopulation\b", question, re.I))
        if sample != population:
            unwanted = "population_" if sample else "sample_"
            result["statistics"] = {
                k: v for k, v in result["statistics"].items() if not k.startswith(unwanted)
            }
        if sample and result["statistics"]["count"] == 1:
            result["undefined"] = (
                "Sample variance and sample standard deviation are undefined for a single observation because n minus 1 is zero."
            )
    return results


def numeric_claim_errors(text):
    """Check explicit English statistic-of-list claims, without guessing intent.

    This is deliberately narrow: prose mathematics is not generally parseable.
    Qualitative reasoning, quartile conventions and formula derivations still
    depend on the tutor model. Questions are not assertions of their answers.
    """
    text = re.sub(r"```[\s\S]*?```|`[^`]*`", "", text)
    text = re.sub(r"[*_]", "", text)
    pattern = re.compile(
        rf"\b(mean|median|range|population variance|sample variance)\s+(?:of|for)\s*"
        rf"(?:the\s+(?:data\s*set|values|observations)\s*)?[\[({{]?({NUMBER_LIST})[\])}}]?"
        rf"\s*(?:is|equals|=|would be)\s*\*{{0,2}}({NUMBER})",
        re.I,
    )
    errors = []
    for match in pattern.finditer(text):
        try:
            stats = describe([float(s) for s in re.findall(NUMBER, match[2])])
            expected = stats[match[1].lower().replace(" ", "_")]
            claimed = float(match[3])
        except (ValueError, KeyError):
            continue
        if not math.isclose(claimed, expected, rel_tol=0.005, abs_tol=1e-8):
            errors.append({"claim": match[0], "computed_result": expected, "statistics": stats})
    return errors


def direct_calculation(question):
    """Answer explicit numeric requests without language-model arithmetic."""
    if re.search(
        r"\b(?:why|who|when|deadline|report|exam|quartile|percentile|interpret|compare|comparison|distribution|bias|assumption|what if|would|should|could|effect|affect|change|sensitive|robust|one[- ]sentence|two[- ]sentences)\b",
        question,
        re.I,
    ):
        return None
    if not re.search(
        r"\b(?:what|find|calculate|compute|evaluate)\b", question, re.I
    ) and not re.match(
        r"\s*(?:median|mean|mode|modes|variance|standard deviation|range)\s+(?:of|for|:)\s*[-+\d.]",
        question,
        re.I,
    ):
        return None
    try:
        calculations = verified_calculations(question)
    except (ValueError, SyntaxError, ArithmeticError):
        return None
    if (
        not calculations
        or any("error" in r for r in calculations)
        or sum("statistics" in r for r in calculations) > 1
    ):
        return None

    def fmt(value):
        if type(value) is int or (
            isinstance(value, float) and value.is_integer() and abs(value) <= 1e15
        ):
            return str(int(value))
        return f"{value:.15g}"

    parts = []
    for calculation in calculations:
        if "expression" in calculation:
            parts.append(
                f"Calculation: {calculation['expression']} = **{fmt(calculation['result'])}**."
            )
            continue
        stats = calculation["statistics"]
        parts.append("Sorted observations: " + ", ".join(fmt(v) for v in stats["sorted"]) + ".")
        if calculation.get("undefined"):
            parts.append(calculation["undefined"])
        if "median" in stats:
            middle = stats["middle_values"]
            step = (
                f"({fmt(middle[0])} + {fmt(middle[1])}) / 2" if len(middle) == 2 else fmt(middle[0])
            )
            parts.append(
                f"**Median: {fmt(stats['median'])}.** "
                + (
                    f"There are {stats['count']} observations, so average the two middle values: {step} = {fmt(stats['median'])}."
                    if len(middle) == 2
                    else "It is the single middle value."
                )
            )
        if "mean" in stats:
            parts.append(
                f"**Mean: {fmt(stats['mean'])}.** Sum divided by count: {fmt(stats['sum'])} / {stats['count']} = {fmt(stats['mean'])}."
            )
        if "modes" in stats:
            values = stats["values"]
            if max(values.count(v) for v in values) == 1:
                parts.append("Every value occurs once, so there is **no unique mode**.")
            else:
                parts.append(
                    "**Mode(s): "
                    + ", ".join(fmt(v) for v in sorted(stats["modes"]))
                    + ".** These values have the highest frequency."
                )
        for name in (
            "range",
            "population_variance",
            "sample_variance",
            "population_standard_deviation",
            "sample_standard_deviation",
        ):
            if name in stats:
                parts.append(f"**{name.replace('_', ' ').capitalize()}: {fmt(stats[name])}.**")
    return "\n\n".join(parts), calculations
