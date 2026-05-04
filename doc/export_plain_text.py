from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "main.tex"
TARGET = ROOT / "main.txt"


SECTION_COMMANDS = {
    "section": "# ",
    "subsection": "## ",
    "subsubsection": "### ",
}


def strip_outer_braces(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text[1:-1].strip()
    return text


def replace_simple_commands(text: str) -> str:
    replacements = {
        r"\textendash": "-",
        r"\_": "_",
        r"\%": "%",
        r"\&": "&",
        r"\lambda": "lambda",
        r"\odot": "odot",
        r"\ast": "*",
        r"\times": "x",
        "~": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_inline_latex(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\hat\{([^{}]*)\}", r"\1_hat", text)
        text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1) / (\2)", text)
        text = re.sub(r"\\texttt\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\emph\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\cite\{([^{}]*)\}", "", text)
        text = re.sub(r"\\ref\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\label\{([^{}]*)\}", "", text)
        text = re.sub(r"\\caption\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\author\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\title\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\date\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
    text = replace_simple_commands(text)
    text = re.sub(r"\$([^$]+)\$", lambda m: replace_simple_commands(m.group(1)), text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\b[Ff]igure\s+fig:[\w:-]+", "Figure", text)
    text = re.sub(r"\b[Tt]able\s+tab:[\w:-]+", "Table", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_equation(lines: list[str]) -> str:
    equation = " ".join(line.strip() for line in lines)
    equation = clean_inline_latex(equation)
    equation = equation.replace(") / (", ") / (")
    return f"Equation: {equation}" if equation else ""


def extract_title_block(tex: str) -> list[str]:
    title = re.search(r"\\title\{(.+?)\}\s*\\author", tex, re.S)
    author = re.search(r"\\author\{(.+?)\}", tex, re.S)
    date = re.search(r"\\date\{(.+?)\}", tex, re.S)
    output: list[str] = []
    if title:
        output.append(clean_inline_latex(title.group(1)))
    if author:
        output.append(clean_inline_latex(author.group(1)))
    if date:
        output.append(clean_inline_latex(date.group(1)))
    return output


def convert_tex_to_text(tex: str) -> str:
    body_match = re.search(r"\\begin\{document\}(.*)\\printbibliography", tex, re.S)
    if not body_match:
        raise RuntimeError("Could not locate LaTeX document body")

    body = body_match.group(1)
    lines = body.splitlines()
    output = extract_title_block(tex)
    if output:
        output.append("")

    skip_env = None
    equation_lines: list[str] | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue

        if equation_lines is not None:
            if line == r"\]":
                equation_text = format_equation(equation_lines)
                if equation_text:
                    output.append(equation_text)
                output.append("")
                equation_lines = None
            else:
                equation_lines.append(line)
            continue

        if line == r"\[":
            equation_lines = []
            continue

        begin_match = re.match(r"\\begin\{([^}]*)\}", line)
        end_match = re.match(r"\\end\{([^}]*)\}", line)
        if begin_match:
            env = begin_match.group(1)
            if env == "tikzpicture":
                skip_env = env
            continue
        if end_match:
            env = end_match.group(1)
            if skip_env == env:
                skip_env = None
            continue
        if skip_env:
            continue

        if line in {r"\maketitle", r"\tableofcontents", r"\centering", r"\clearpage"}:
            continue
        if line.startswith(r"\includegraphics") or line.startswith(r"\graphicspath"):
            continue
        if line.startswith(r"\resizebox"):
            continue
        if line in {"}", "%", r"\hfill", r"\vspace{0.8em}"}:
            continue

        section_match = re.match(r"\\(section|subsection|subsubsection)\{(.+?)\}", line)
        if section_match:
            prefix = SECTION_COMMANDS[section_match.group(1)]
            output.append(f"{prefix}{clean_inline_latex(section_match.group(2))}")
            output.append("")
            continue

        caption_match = re.match(r"\\caption\{(.+?)\}", line)
        if caption_match:
            output.append(f"Caption: {clean_inline_latex(caption_match.group(1))}")
            output.append("")
            continue

        if "&" in line or r"\\" in line:
            row = line.replace(r"\\", "")
            row = row.replace("&", " | ")
            row = clean_inline_latex(row)
            if row and row not in {"@lccccccc@", "@lcc@"}:
                output.append(row)
            continue

        cleaned = clean_inline_latex(line)
        if cleaned:
            output.append(cleaned)

    while output and output[-1] == "":
        output.pop()

    return "\n".join(output) + "\n"


def main() -> None:
    tex = SOURCE.read_text(encoding="utf-8")
    TARGET.write_text(convert_tex_to_text(tex), encoding="utf-8")


if __name__ == "__main__":
    main()